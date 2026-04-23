import argparse
import optuna

parser = argparse.ArgumentParser()
parser.add_argument("--storage", required=True)
parser.add_argument("--study-name", required=True)
args = parser.parse_args()

study = optuna.load_study(
    study_name=args.study_name,
    storage=args.storage,
)

best = study.best_trial

print(f"\n=== Best Trial ===")
print(f"  Trial number : {best.number}")
print(f"  PR-AUC       : {best.value:.4f}")
print(f"  Threshold    : {best.user_attrs.get('best_threshold', 'N/A')}")
print(f"\n  Params:")
for k, v in best.params.items():
    print(f"    {k}: {v}")

print(f"\n  Add to config.py:")
print(f"  CONFIG = ExperimentConfig(")
print(f"      model=CNNTransformerConfig(")
print(f"          d_model={best.params['d_model']},")
print(f"          num_heads={best.params['nhead']},")
print(f"          num_layers={best.params['num_layers']},")
print(f"          dropout={best.params['dropout']:.3f},")
print(f"          pos_weight_cap={best.params['pos_weight_cap']:.2f},")
print(f"      ),")
print(f"      training=TrainingConfig(")
print(f"          learning_rate={best.params['lr']:.2e},")
print(f"          weight_decay={best.params['weight_decay']:.2e},")
print(f"          batch_size={best.params['batch_size']},")
print(f"      ),")
print(f"  )")
