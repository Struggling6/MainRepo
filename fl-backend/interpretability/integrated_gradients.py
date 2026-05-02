from __future__ import annotations

from pathlib import Path

import torch
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader

from config import CONFIG, ExperimentConfig
from data.registry import create_dataset_handler
from interpretability.attribution_utils import (
    get_device,
    load_checkpoint_model,
    save_npz,
    summarize_attributions,
)
from interpretability.baselines import make_baseline


class IntegratedGradientsRunner:
    def __init__(self, config: ExperimentConfig = CONFIG):
        self.config = config
        self.ig_config = config.interpretability
        self.device = get_device()

    def run(
        self,
        partition_id: int = 0,
        max_batches: int | None = 1,
        target_class: int | None = None,
    ) -> Path:
        handler = create_dataset_handler(self.config)
        metadata = handler.get_metadata()

        trainloader, testloader = handler.get_dataloaders(
            partition_id=partition_id
        )

        model, checkpoint = load_checkpoint_model(
            checkpoint_path=self.config.evaluation.model_path,
            config=self.config,
            metadata=metadata,
            device=self.device,
        )

        attributions, inputs, labels, logits, deltas = self.attribute_loader(
            model=model,
            loader=testloader,
            reference_loader=trainloader,
            max_batches=max_batches,
            target_class=target_class,
        )

        summaries = summarize_attributions(attributions)

        output_path = self.ig_config.ig_output_path

        save_npz(
            output_path,
            **summaries,
            inputs=inputs,
            labels=labels,
            logits=logits,
            convergence_delta=deltas,
            feature_names=getattr(handler, "feature_cols", None),
            metadata=metadata,
            model_name=self.config.model.name,
            baseline=self.ig_config.ig_baseline,
            ig_steps=self.ig_config.ig_steps,
            checkpoint_threshold=checkpoint.get("threshold"),
            checkpoint_metrics=checkpoint.get("metrics", {}),
        )

        print(f"Integrated Gradients saved to: {output_path}")
        return output_path

    def attribute_loader(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        reference_loader: DataLoader | None = None,
        max_batches: int | None = None,
        target_class: int | None = None,
    ):
        reference_batch = None

        if reference_loader is not None:
            reference_batch, _ = next(iter(reference_loader))
            reference_batch = reference_batch.to(self.device)

        all_attr = []
        all_inputs = []
        all_labels = []
        all_logits = []
        all_deltas = []

        for batch_idx, (inputs, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            baseline = make_baseline(
                inputs=inputs,
                mode=self.ig_config.ig_baseline,
                reference_batch=reference_batch,
            )

            attr, logits, delta = self.integrated_gradients(
                model=model,
                inputs=inputs,
                baselines=baseline,
                target_class=target_class,
            )

            all_attr.append(attr.cpu())
            all_inputs.append(inputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_logits.append(logits.detach().cpu())
            all_deltas.append(delta.cpu())

        return (
            torch.cat(all_attr, dim=0),
            torch.cat(all_inputs, dim=0),
            torch.cat(all_labels, dim=0),
            torch.cat(all_logits, dim=0),
            torch.cat(all_deltas, dim=0),
        )

    def integrated_gradients(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        baselines: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model.eval()

        ig = IntegratedGradients(model)

        with torch.no_grad():
            logits = model(inputs)

        if target_class is None:
            if logits.ndim == 2 and logits.shape[1] > 1:
                target_class = int(torch.argmax(logits.mean(dim=0)).item())
            else:
                target_class = None

        attributions, delta = ig.attribute(
            inputs=inputs,
            baselines=baselines,
            target=target_class,
            n_steps=self.ig_config.ig_steps,
            return_convergence_delta=True,
        )

        return attributions.detach(), logits.detach(), delta.detach()


def run_integrated_gradients(
    config: ExperimentConfig = CONFIG,
    partition_id: int = 0,
    max_batches: int | None = 1,
    target_class: int | None = None,
) -> Path:
    return IntegratedGradientsRunner(config).run(
        partition_id=partition_id,
        max_batches=max_batches,
        target_class=target_class,
    )