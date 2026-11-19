import kagglehub


DATASET = "valakhorasani/electric-vehicle-charging-patterns"


def main() -> None:
    try:
        path = kagglehub.dataset_download(DATASET)
    except Exception as exc:
        raise RuntimeError(
            f"Impossible de télécharger le jeu de données Kaggle '{DATASET}'. "
            "Vérifiez votre connexion Internet."
        ) from exc

    print("Chemin des fichiers du jeu de données :", path)


if __name__ == "__main__":
    main()