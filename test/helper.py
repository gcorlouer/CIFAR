def CIFAR_filename(sub_id="sub-00", task="rest", run="01",
                   ext=".set"):
    """Return filename given subject id, task, run, and datatype """
    dataset = [sub_id,  task, "baseline", f"run-{run}", 'ieeg']
    dataset = "_".join(dataset)
    dataset = dataset + ext
    return dataset
