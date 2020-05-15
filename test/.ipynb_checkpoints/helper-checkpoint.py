from pathlib import Path, PurePath

def BIDS_filename(sub_id="sub-00", task="rest", run="01",
                   ext=".set"):
    """Return filename given subject id, task, run, and datatype """
    dataset = [sub_id,  task, "baseline", f"run-{run}", 'ieeg']
    dataset = "_".join(dataset)
    dataset = dataset + ext
    return dataset

def CIFAR_filename(subid="JuRo", task="sleep", run="1", proc="raw",
                      suffix=".set"):
    """Return filename given subject id, task, run, and datatype """
    if task == 'sleep':
        if proc == 'raw':
            dataset = [subid, "freerecall", task, 'preprocessed']
            dataset = "_".join(dataset)
            dataset = dataset + suffix
        else:
            dataset = [subid, "freerecall", task, 'preprocessed', 'BP', 'montage']
            dataset = "_".join(dataset)
            dataset = dataset + suffix
    else:
        if proc == 'raw':
            dataset = [subid, "freerecall", task, run, 'preprocessed']
            dataset = "_".join(dataset)
            dataset = dataset + suffix
        else:
            dataset = [subid, "freerecall", task, run, 'preprocessed', 'BP', 'montage']
            dataset = "_".join(dataset)
            dataset = dataset + suffix       
    return dataset

def subject_path(data_dir=Path('~','CIFAR_data').expanduser(), subid='JuRo', 
                 montage='bipolar_montage'):
    
    cfsubdir = data_dir.joinpath('iEEG_10','subjects', subid)
    data_subdir = cfsubdir.joinpath('EEGLAB_datasets', montage)
    return cfsubir, data_subdir