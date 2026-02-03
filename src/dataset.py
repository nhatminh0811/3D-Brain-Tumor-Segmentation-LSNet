import os
from glob import glob


def get_brats2020_datalist(root_dir, section="training"):
    """
    Create MONAI-style datalist for BraTS 2020
    """
    if section == "training":
        data_dir = os.path.join(root_dir, "TrainingData")
    else:
        data_dir = os.path.join(root_dir, "ValidationData")

    # only include directories (skip CSV files like name_mapping.csv)
    subjects = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    datalist = []

    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)

        # expected names, but allow fallback to glob patterns if exact names differ
        def find_modality(mod):
            # check common exact filenames
            for ext in (".nii.gz", ".nii"):
                exact = os.path.join(subject_dir, f"{subject}_{mod}{ext}")
                if os.path.exists(exact):
                    return exact
            # try glob patterns (case-insensitive by checking lower/upper)
            patterns = [
                os.path.join(subject_dir, f"*{mod}*.nii.gz"),
                os.path.join(subject_dir, f"*{mod}*.nii"),
                os.path.join(subject_dir, f"*{mod.upper()}*.nii.gz"),
                os.path.join(subject_dir, f"*{mod.upper()}*.nii"),
            ]
            for pat in patterns:
                matches = sorted(glob(pat))
                if matches:
                    return matches[0]
            # fallback: return the expected .nii (so missing check will catch it)
            return os.path.join(subject_dir, f"{subject}_{mod}.nii")

        image = [
            find_modality('flair'),
            find_modality('t1'),
            find_modality('t1ce'),
            find_modality('t2'),
        ]

        item = {"image": image}

        if section == "training":
            item["label"] = os.path.join(
                subject_dir, f"{subject}_seg.nii.gz"
            )

        # check all files exist; if not, skip subject and warn
        expected_files = list(image)
        if section == "training":
            # label may be named differently; try fallback
            label_exact = os.path.join(subject_dir, f"{subject}_seg.nii.gz")
            if os.path.exists(label_exact):
                item["label"] = label_exact
            else:
                label_matches = sorted(glob(os.path.join(subject_dir, "*seg*.nii*")))
                if label_matches:
                    item["label"] = label_matches[0]
                else:
                    expected_files.append(label_exact)

        missing = [p for p in expected_files if not os.path.exists(p)]
        if missing:
            print(f"Skipping {subject}: missing files: {missing}")
            continue

        datalist.append(item)

    return datalist
