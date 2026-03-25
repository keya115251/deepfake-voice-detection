import os
from utils.dataset_loader import get_file_paths


def main():

    audio_dir = os.path.join(
        "data",
        "processed",
        "ASVspoof2019",
        "LA",
        "train"
    )

    protocol_path = os.path.join(
        "data",
        "raw",
        "ASVspoof2019",
        "LA",
        "ASVspoof2019_LA_cm_protocols",
        "ASVspoof2019.LA.cm.train.trn.txt"
    )

    print("Checking paths...\n")

    if not os.path.exists(audio_dir):
        print("❌ Audio directory not found")
        return

    if not os.path.exists(protocol_path):
        print("❌ Protocol file not found")
        return

    print("✅ Paths OK\n")

    files, labels = get_file_paths(audio_dir, protocol_path)

    print("Total files found:", len(files))

    if len(files) > 0:
        print("First file:", files[0])
        print("First label:", labels[0])

        print("\nClass Distribution:")
        print("Real (0):", labels.count(0))
        print("Fake (1):", labels.count(1))


if __name__ == "__main__":
    main()