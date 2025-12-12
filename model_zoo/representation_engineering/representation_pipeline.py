#!/usr/bin/env python3
import torch
import logging
import json
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .data_loader import RepresentationDatasetLoader
from .steering_vector_computer import SteeringVectorComputer
from .model_modifier import ModelModifier
from .evaluator import RepresentationEvaluator
from .dataset_utils import RepresentationEngineeringDatasetBuilder

logger = logging.getLogger(__name__)


class RepresentationPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # setup device
        device_type = config.get("device", {}).get("type", "auto")
        if device_type == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device_type

        # setup run directory
        run_dir = Path(config.get("run_dir", "results/representation_engineering"))
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir

        logger.info(
            f"Initialized pipeline - Device: {self.device}, Run dir: {self.run_dir}"
        )

    def run(self):
        try:
            # parse dataset config
            steering_dataset = self.config["steering"]["steering_vector_dataset"]
            subject_dataset = self.config["dataset"]["subject_model_dataset"]
            logger.info(
                f"Two-dataset mode - Steering: {steering_dataset}, Subject: {subject_dataset}"
            )

            # phase 1: load steering dataset and compute vectors
            logger.info("\n" + "=" * 70)
            logger.info("PHASE 1: steering vector computation")
            logger.info("=" * 70)

            # extract input mode config
            input_mode = self.config["encoder_decoder"].get("input_mode", "weights")
            max_dimensions = (
                self.config["encoder_decoder"].get("max_dimensions")
                if input_mode in ["signature", "both"]
                else None
            )
            method_names = (
                self.config["encoder_decoder"].get("neuron_profile", {}).get("methods", [])
                if input_mode in ["signature", "both"]
                else None
            )

            steering_loader = RepresentationDatasetLoader(
                hf_dataset_path=steering_dataset,
                encoder_repo_id=self.config["encoder_decoder"]["encoder_repo_id"],
                decoder_repo_id=self.config["encoder_decoder"].get("decoder_repo_id"),
                input_mode=input_mode,
                max_dimensions=max_dimensions,
                method_names=method_names,
                device=self.device,
                max_models=self.config["steering"].get("max_models_steering"),
            )

            steering_models_data = steering_loader.load_and_encode_models()
            all_patterns = steering_loader.get_all_patterns()
            logger.info(f"Discovered {len(all_patterns)} patterns: {all_patterns}")

            pattern_clusters = steering_loader.group_by_patterns()

            steering_computer = SteeringVectorComputer(
                pattern_clusters=pattern_clusters,
                device=self.device,
                cache_dir=Path(self.config["steering"]["cache_dir"]),
                normalize_vectors=self.config["steering"].get(
                    "normalize_vectors", False
                ),
            )

            cache_metadata = {
                "steering_dataset_path": steering_dataset,
                "encoder_repo_id": self.config["encoder_decoder"]["encoder_repo_id"],
                "latent_dim": steering_loader.latent_dim,
                "normalize_vectors": self.config["steering"].get(
                    "normalize_vectors", False
                ),
                "n_models": len(steering_models_data),
            }

            if self.config["steering"].get("compute_on_init", True):
                force_recompute = self.config["steering"].get("force_recompute", False)
                if not force_recompute:
                    cache_loaded = steering_computer.load_cache(
                        expected_metadata=cache_metadata,
                        validate=self.config["steering"].get("validate_cache", True),
                    )
                else:
                    cache_loaded = False

                if cache_loaded:
                    logger.info("Loaded steering vectors from cache")
                else:
                    logger.info("Computing pairwise steering vectors...")
                    steering_computer.compute_all_pairwise_vectors()
                    # log sample of transformations
                    logged_count = 0
                    for from_pattern in sorted(all_patterns):
                        if logged_count >= 5:  # limit logging
                            break
                        for to_pattern in sorted(all_patterns):
                            if from_pattern == to_pattern:
                                continue
                            try:
                                stats = steering_computer.get_vector_statistics(from_pattern, to_pattern)
                                logger.info(
                                    f"  {from_pattern}→{to_pattern}: norm={stats['norm']:.4f}, "
                                    f"n_from={stats['n_from']}, n_to={stats['n_to']}"
                                )
                                logged_count += 1
                                if logged_count >= 5:
                                    break
                            except:
                                pass
                    steering_computer.save_cache(metadata=cache_metadata)

            # free memory
            del steering_loader
            del steering_models_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # phase 2: load subject models
            logger.info("\n" + "=" * 70)
            logger.info("PHASE 2: loading subject models")
            logger.info("=" * 70)

            subject_loader = RepresentationDatasetLoader(
                hf_dataset_path=subject_dataset,
                encoder_repo_id=self.config["encoder_decoder"]["encoder_repo_id"],
                decoder_repo_id=self.config["encoder_decoder"].get("decoder_repo_id"),
                input_mode=input_mode,
                max_dimensions=max_dimensions,
                method_names=method_names,
                device=self.device,
                max_models=self.config["dataset"].get("max_models_subject"),
            )

            subject_models_data = subject_loader.load_and_encode_models()
            subject_models = subject_loader.get_subject_models(
                sample_size=self.config["dataset"].get("sample_size")
            )
            logger.info(f"Selected {len(subject_models)} subject models")

            # phase 3: validate pattern coverage
            logger.info("\n" + "=" * 70)
            logger.info("PHASE 3: Validating pattern coverage")
            logger.info("=" * 70)

            subject_patterns = set(subject_loader.get_all_patterns())
            steering_patterns = set(all_patterns)
            missing_patterns = subject_patterns - steering_patterns
            if missing_patterns:
                raise ValueError(
                    f"Missing steering vectors for patterns: {missing_patterns}"
                )
            logger.info(f"All {len(subject_patterns)} subject patterns covered")

            # phase 4 & 5: apply modifications and evaluate
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 4 & 5: Modifications and evaluation")
            logger.info("=" * 60)

            modifier = ModelModifier(
                encoder=subject_loader.encoder,
                decoder=subject_loader.decoder,
                tokenizer=subject_loader.tokenizer,
                input_mode=input_mode,
                max_dims=max_dimensions,
                method_names=method_names,
                device=self.device,
            )

            evaluator = None
            if self.config["evaluation"].get("enabled", True):
                evaluator = RepresentationEvaluator(
                    benchmark_dataset_path=self.config["evaluation"][
                        "benchmark_dataset_path"
                    ],
                    device=self.device,
                    batch_size=self.config["evaluation"].get("batch_size", 32),
                )

            all_results = []
            input_weights_dict = {}
            result_counter = 0

            # for each subject model, apply pairwise transformations
            for model_id, (original_idx, original_weights, metadata, signature) in enumerate(subject_models):
                # get this model's latent vector from subject_models_data using original index
                _, _, subject_latent = subject_models_data[original_idx]

                # get full pattern list from metadata for evaluation contamination filtering
                source_patterns = metadata.get("selected_patterns", [])
                if not source_patterns:
                    # fallback if metadata missing
                    source_patterns = [steering_computer.find_nearest_pattern(subject_latent)]

                # detect nearest pattern for steering vector selection (current behavior maintained)
                detected_pattern = steering_computer.find_nearest_pattern(subject_latent)

                logger.info(
                    f"\nModel {model_id + 1}/{len(subject_models)}: patterns={source_patterns}, detected={detected_pattern}"
                )
                input_weights_dict[original_idx] = original_weights

                transformations_tried = 0
                # try transforming to each other pattern
                for target_pattern in all_patterns:
                    # skip if target is already in source patterns
                    if target_pattern in source_patterns:
                        continue

                    transformations_tried += 1
                    logger.info(
                        f"  [{transformations_tried}] {detected_pattern} → {target_pattern}"
                    )

                    # get pairwise steering vector using detected pattern
                    steering_vector = steering_computer.get_steering_vector(
                        detected_pattern, target_pattern
                    )

                    # apply modification
                    modified_weights = modifier.modify_model(
                        subject_weights=original_weights,
                        subject_signature=signature,
                        steering_vectors=[(f"{detected_pattern}→{target_pattern}", steering_vector)],
                        strength=self.config["modification"].get("strength", 1.0),
                    )

                    # save modified weights
                    if self.config["logging"].get("save_modified_weights", True):
                        weights_dir = self.run_dir / "modified_weights"
                        weights_dir.mkdir(exist_ok=True)
                        torch.save(
                            modified_weights,
                            weights_dir
                            / f"model_{original_idx}_result_{result_counter}.pt",
                        )

                    # evaluate
                    if evaluator:
                        eval_results = evaluator.evaluate_modification(
                            original_weights=original_weights,
                            modified_weights=modified_weights,
                            original_patterns=source_patterns,  # full pattern list from metadata
                            all_patterns=all_patterns,
                            metadata=metadata,
                        )

                        all_results.append(
                            {
                                "model_id": original_idx,
                                "result_id": result_counter,
                                "source_patterns": source_patterns,  # list of all source patterns
                                "detected_pattern": detected_pattern,  # pattern used for steering
                                "target_pattern": target_pattern,
                                "transformation": f"{detected_pattern}→{target_pattern}",
                                "strength": self.config["modification"].get(
                                    "strength", 1.0
                                ),
                                "evaluation": eval_results,
                            }
                        )
                        result_counter += 1

                logger.info(f"  Completed {transformations_tried} transformations")

            # phase 6: save results
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 6: Saving results")
            logger.info("=" * 60)

            # save results
            if self.config["logging"].get("save_evaluation_results", True):
                results_path = self.run_dir / "evaluation_results.json"
                with open(results_path, "w") as f:
                    json.dump(self._make_json_serializable(all_results), f, indent=2)
                logger.info(f"Saved evaluation results")

            config_path = self.run_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved config")

            # generate report
            report_path = self.run_dir / "report.md"
            with open(report_path, "w") as f:
                f.write("# Representation Engineering Report\n\n")
                f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
                f.write(f"**Run Directory:** `{self.run_dir}`\n\n")
                f.write(f"- **Models Evaluated:** {len(all_results)}\n\n")

                if evaluator and all_results:
                    f.write("## Aggregate Statistics\n\n")
                    aggregate = evaluator.compute_aggregate_statistics(
                        [r["evaluation"] for r in all_results if "evaluation" in r]
                    )
                    if aggregate:
                        f.write(f"- **Evaluations:** {aggregate['n_evaluations']}\n")
                        f.write(f"- **Failed:** {aggregate['n_failed']}\n\n")
                        f.write(f"### Macro-Averaged Deltas\n\n")
                        f.write(
                            f"- **Mean F1 Delta:** {aggregate['mean_macro_f1_delta']:+.3f}\n"
                        )
                        f.write(
                            f"- **Mean Accuracy Delta:** {aggregate['mean_macro_accuracy_delta']:+.3f}\n\n"
                        )
                        f.write(f"### Micro-Averaged Deltas\n\n")
                        f.write(
                            f"- **Mean F1 Delta:** {aggregate['mean_micro_f1_delta']:+.3f}\n"
                        )
                        f.write(
                            f"- **Mean Accuracy Delta:** {aggregate['mean_micro_accuracy_delta']:+.3f}\n\n"
                        )
                        if aggregate.get("pattern_f1_delta_means"):
                            f.write(f"### Per-Pattern F1 Delta Statistics\n\n")
                            for pattern in sorted(aggregate["pattern_f1_delta_means"].keys()):
                                mean = aggregate["pattern_f1_delta_means"][pattern]
                                std = aggregate["pattern_f1_delta_stds"][pattern]
                                f.write(f"- **{pattern}:** {mean:+.3f} (±{std:.3f})\n")
                            f.write("\n")
            logger.info(f"Generated report")

            # phase 7: upload to huggingface
            if self.config.get("huggingface", {}).get("enabled", False):
                logger.info("\n" + "=" * 60)
                logger.info("PHASE 7: Uploading to HuggingFace")
                logger.info("=" * 60)

                hf_config = self.config.get("huggingface", {})
                dataset_builder = RepresentationEngineeringDatasetBuilder(self.run_dir)
                dataset_builder.add_from_results(
                    results=all_results, input_weights_dict=input_weights_dict
                )

                if hf_config.get("save_local", True):
                    local_dataset_dir = self.run_dir / "huggingface_dataset"
                    dataset_builder.save_local(local_dataset_dir)
                    logger.info(f"Saved dataset locally")

                repo_id = hf_config.get("repo_id")
                if repo_id:
                    try:
                        dataset_builder.upload_to_hub(
                            repo_id=repo_id,
                            private=hf_config.get("private", False),
                            token=hf_config.get("token"),
                        )
                        logger.info("Uploaded to HuggingFace Hub")
                    except Exception as e:
                        logger.error(f"Failed to upload: {e}", exc_info=True)
                else:
                    logger.error("HuggingFace repo_id not specified")

            logger.info("\n" + "=" * 60)
            logger.info(f"COMPLETE - Results: {self.run_dir}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def _make_json_serializable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)
