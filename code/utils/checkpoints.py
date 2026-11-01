"""Check Stage 1 training metadata before starting Stage 2."""


def validate_stage1_checkpoint(checkpoint, *, bank_size, seed, encode_knn):
    config = checkpoint.get("config", {})
    expected = {
        "stage": 1,
        "bank_size": bank_size,
        "negative_mining": "curv_hard",
        "curv_thresh": 0.05,
        "curv_fallback_thresh": 0.10,
        "margin": 0.3,
        "margin_scale": 0.1,
        "right_triplet_weight": 0.5,
        "environment": "score-denoise",
        "seed": seed,
    }
    version = config.get("format_version")
    if version == 1:
        expected.update(
            model="GeoTripletDenoise",
            encode_knn=encode_knn,
            negative_candidates="filled_entries",
            stage1_positive="once_noisy",
        )
    elif version is None:
        # Accept compatible checkpoints saved before the metadata cleanup.
        # These fields are checked on input only and are not emitted by training.
        expected.update(
            pretrain_mode="dual",
            stage1_main_positive="noisypos",
            noisypos_aux_weight=0.0,
            local_aux_weight=0.0,
            bank_policy="filled_only",
        )
    else:
        raise ValueError("encoder_ckpt has an unsupported checkpoint format version")
    mismatches = [key for key, value in expected.items() if config.get(key) != value]
    if mismatches:
        raise ValueError("encoder_ckpt has incompatible Stage 1 settings: " + ", ".join(mismatches))
