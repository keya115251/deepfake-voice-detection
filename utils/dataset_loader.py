import os


def parse_protocol(protocol_path):
    """
    Reads the ASVspoof protocol file and creates a dictionary:
    { file_id : label }

    label:
        0 -> bonafide (real)
        1 -> spoof (fake)
    """

    labels = {}

    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            # Format:
            # speaker_id file_id - attack_type label
            file_id = parts[1]
            label = parts[-1]

            if label == "bonafide":
                labels[file_id] = 0
            else:
                labels[file_id] = 1

    return labels


def get_file_paths(audio_dir, protocol_path):
    """
    Matches wav files with labels using the protocol file.
    Returns:
        files  -> list of file paths
        labels -> corresponding list of 0/1 labels
    """

    labels_dict = parse_protocol(protocol_path)

    files = []
    labels = []

    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav"):

            file_id = filename.replace(".wav", "")

            if file_id in labels_dict:
                full_path = os.path.join(audio_dir, filename)

                files.append(full_path)
                labels.append(labels_dict[file_id])

    return files, labels