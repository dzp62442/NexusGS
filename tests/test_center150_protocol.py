import json
import tempfile
import unittest
from pathlib import Path

from comp_svfgs.dataset_omniscene import (
    CENTER150_FILENAME,
    CENTER150_SAMPLE_COUNT,
    load_center150_tokens,
)
from scripts.run_omniscene import (
    ALL_18_VIEW_GROUP,
    EVALUATION_FORMAT_VERSION,
    LEGACY_EVALUATION_FORMAT_VERSION,
    NOVEL_12_VIEW_GROUP,
    aggregate_center150,
    evaluation_has_view_groups,
    load_evaluation,
    load_json,
    scene_complete,
    scene_name,
    write_scene_completion,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class DummyDataset:
    def __init__(self, bin_tokens):
        self.bin_tokens = bin_tokens

    def __len__(self):
        return len(self.bin_tokens)


class Center150ProtocolTest(unittest.TestCase):
    def test_center150_manifest_is_validated_against_val_groups(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            version_dir = Path(temporary_dir)
            tokens = [f"scene{index:04x}_bin0" for index in range(CENTER150_SAMPLE_COUNT)]
            write_json(
                version_dir / "bins_val_3.2m.json",
                {"bins": tokens, "adjacent_bins": [[token] for token in tokens]},
            )
            write_json(version_dir / CENTER150_FILENAME, {"bins": tokens})
            bin_info_dir = version_dir / "bin_infos_3.2m"
            bin_info_dir.mkdir()
            for token in tokens:
                (bin_info_dir / f"{token}.pkl").write_bytes(b"test")

            self.assertEqual(load_center150_tokens(str(version_dir)), tokens)
            write_json(version_dir / CENTER150_FILENAME, {"bins": list(reversed(tokens))})
            with self.assertRaisesRegex(ValueError, "lower-median"):
                load_center150_tokens(str(version_dir))

    def test_completed_scenes_are_checked_and_aggregated(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            output_root = root / "preprocessed"
            results_root = root / "results"
            dataset = DummyDataset(["scenea_bin0", "sceneb_bin0"])
            eval_iterations = [1_000, 5_000, 10_000]
            fingerprint = "protocol-test"

            for index, bin_token in enumerate(dataset.bin_tokens):
                name = scene_name(index, len(dataset), bin_token)
                scene_dir = output_root / name
                model_dir = results_root / name
                image_names = [f"view_{view_index:02d}.png" for view_index in range(18)]
                write_json(
                    scene_dir / "target" / "cameras.json",
                    {"views": [{"image_name": Path(name).stem} for name in image_names]},
                )
                for iteration in eval_iterations:
                    (model_dir / "point_cloud" / f"iteration_{iteration}").mkdir(
                        parents=True, exist_ok=True
                    )
                    (model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply").write_bytes(b"ply")
                    (model_dir / f"chkpnt{iteration}.pth").write_bytes(b"checkpoint")
                    for subdir in ("renders", "gt"):
                        image_dir = model_dir / "test" / f"ours_{iteration}" / subdir
                        image_dir.mkdir(parents=True, exist_ok=True)
                        for image_name in image_names:
                            (image_dir / image_name).write_bytes(b"png")
                    write_json(
                        model_dir / "evaluation" / f"iteration_{iteration}.json",
                        {
                            "format_version": EVALUATION_FORMAT_VERSION,
                            "iteration": iteration,
                            "split": "test",
                            "num_views": len(image_names),
                            "metrics": (all_metrics := {
                                "psnr": float(index + iteration / 1_000),
                                "ssim": 0.8 + index * 0.1,
                                "lpips": 0.2 - index * 0.1,
                                "l1": 0.1,
                            }),
                            "view_groups": {
                                ALL_18_VIEW_GROUP: {
                                    "num_views": len(image_names),
                                    "metrics": dict(all_metrics),
                                },
                                NOVEL_12_VIEW_GROUP: {
                                    "num_views": 12,
                                    "metrics": {
                                        **all_metrics,
                                        "psnr": all_metrics["psnr"] + 1.0,
                                    },
                                },
                            },
                            "training_time_seconds": float(iteration / 10 + index),
                        },
                    )
                write_scene_completion(
                    scene_dir,
                    model_dir,
                    index + 1,
                    bin_token,
                    eval_iterations,
                    fingerprint,
                )
                self.assertTrue(
                    scene_complete(
                        scene_dir,
                        model_dir,
                        eval_iterations,
                        fingerprint,
                        bin_token,
                    )
                )

            self.assertTrue(
                aggregate_center150(
                    dataset,
                    output_root,
                    results_root,
                    eval_iterations,
                    fingerprint,
                )
            )
            summary = load_json(results_root / "center150_metrics_summary.json")
            self.assertEqual(summary["num_samples"], 2)
            self.assertAlmostEqual(summary["averages"]["1000"]["psnr"], 1.5)
            self.assertAlmostEqual(
                summary["averages"]["1000"]["view_groups"][NOVEL_12_VIEW_GROUP][
                    "psnr"
                ],
                2.5,
            )
            self.assertAlmostEqual(
                summary["averages"]["10000"]["training_time_seconds"], 1000.5
            )

            first_name = scene_name(0, len(dataset), dataset.bin_tokens[0])
            broken_render = (
                results_root / first_name / "test" / "ours_1000" / "renders" / "view_00.png"
            )
            broken_render.unlink()
            self.assertFalse(
                scene_complete(
                    output_root / first_name,
                    results_root / first_name,
                    eval_iterations,
                    fingerprint,
                    dataset.bin_tokens[0],
                )
            )

    def test_legacy_evaluation_is_accepted_but_needs_view_group_backfill(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            model_dir = Path(temporary_dir)
            training_time = 12.3456789
            write_json(
                model_dir / "evaluation" / "iteration_1000.json",
                {
                    "format_version": LEGACY_EVALUATION_FORMAT_VERSION,
                    "iteration": 1_000,
                    "split": "test",
                    "num_views": 18,
                    "metrics": {
                        "psnr": 30.0,
                        "ssim": 0.8,
                        "lpips": 0.2,
                        "l1": 0.05,
                    },
                    "training_time_seconds": training_time,
                },
            )

            record = load_evaluation(model_dir, 1_000, 18)
            self.assertFalse(evaluation_has_view_groups(record))
            self.assertEqual(record["training_time_seconds"], training_time)


if __name__ == "__main__":
    unittest.main()
