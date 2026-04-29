from .base import BaseModel
from pypots.classification import TimesNet as PyPOTSTimesNet

class TimesNetModel(BaseModel):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_classes: int,
        n_layers: int,
        top_k: int,
        d_model: int,
        d_ffn: int,
        n_kernels: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # PyPOTS wrapper. Vi bruger kun dens interne torch-model.
        pypots_model = PyPOTSTimesNet(
            n_steps=n_steps,
            n_features=n_features,
            n_classes=n_classes,
            n_layers=n_layers,
            top_k=top_k,
            d_model=d_model,
            d_ffn=d_ffn,
            n_kernels=n_kernels,
            dropout=dropout,

            # Disse er mest relevante for PyPOTS' egen .fit().
            # Din egen training loop bruger dem typisk ikke.
            batch_size=32,
            epochs=1,
            patience=None,
            saving_path=None,
            model_saving_strategy=None,
            verbose=False,
        )

        # Dette er den rigtige torch.nn.Module-del.
        self.model = pypots_model.model

    def forward(self, x):
        """
        x shape: [batch_size, n_steps, n_features]
        """
        outputs = self.model({"X": x})
        logits = outputs["logits"]

        # For BCEWithLogitsLoss vil du normalt have shape [batch_size]
        # når n_classes = 1.
        if logits.shape[-1] == 1:
            return logits.squeeze(-1)

        # For CrossEntropyLoss / multiclass beholdes shape [batch_size, n_classes]
        return logits