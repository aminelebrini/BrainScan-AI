# BrainScan-AI

## Comparer le feature extraction et le fine-tuning

Les historiques Keras présents dans `results/feature_extraction/` et
`results/fin-tuining/` peuvent être comparés avec Matplotlib et Seaborn :

```bash
python main.py
```

Après l'entraînement, `main.py` instancie `Visualisation` et appelle
`display_training_results()`. La méthode affiche les courbes et crée :

- `training_comparison.png` : courbes d'accuracy et de loss pour l'entraînement
  et la validation ;
