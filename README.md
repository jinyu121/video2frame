# video2frame

Video2frame is also an easy-to-use tool to extract frames from video.

## Why this tool

[Forwchen's vid2frame tool](https://github.com/forwchen/vid2frame) is great, but I am always confused by their parameters. At the same time, I also want to add something I need to the tool. 

So I re-wrote the code. And now, it is a new wheel. It is hard to make a PR since I changed the code style. 

## How to use

1. ### Establish the environment
    
    We recommend using [conda](https://conda.io/) to establish the environment. Just using 
    
    ```sh
    conda env create -f install/conda-environment.yml
    ```
    
    You can also do it manually. This project relays on the following packages:
    
    - Python
    - FFmpeg
    - Python packages (can be installed using `pip install -r install/pip-requirements.txt`)
        + h5py
        + lmdb
        + numpy
        + easydict
        + tqdm
    
1. ### Make the annotation json file
    
    The json file should like
    
    ```json
    {
        "meta": {
            "class_num": 2,
            "class_name": [
                "class_1",
                "class_2"
            ]
        },
        "annotation": {
            "label1_abcdefg": {
                "path": "path/to/the/video/file_1.mp4",
                "class": 1
            },
            "label2_asdfghj": {
                "path": "path/to/the/video/file_2.mp4",
                "class": 2
            }
        }
    }
    ```
    
1. ### Extract frames using `video2frame.py`
    
    #### Examples
    
    + Using the default options:
     
        ```sh
        python video2frame.py dataset.json
        ```
        
    + Specify the output file name:
    
        ```sh
        python video2frame.py dataset.json --db_name my_dataset
        ```
        
    + Using lmdb rather than hdf5:
    
        ```sh
        python video2frame.py dataset.json --db_type LMDB
        ``` 
        or 
        ```sh
        python video2frame.py dataset.json --db_name my_dataset.lmdb
        ```
        
    + Random clip 5 seconds:
    
        ```sh
        python video2frame.py dataset.json --duration 5.0
        ```
        
    + Get 3 video clips with a length of 5 seconds:
    
        ```sh
        python video2frame.py dataset.json --clips 3 --duration 5.0 
        ```
    
    + Resize the frames to 320x240:
        
        ```sh
        python video2frame.py dataset.json --resize_mode 1 --resize 320x240
        ```
    
    + Keep the aspect ration, and resize the shorter side to 320:
    
        ```sh
        python video2frame.py dataset.json --resize_mode 2 --resize S320
        ```
    
    + Keep the aspect ration, and resize the longer side to 240:
    
        ```sh
        python video2frame.py dataset.json --resize_mode 2 --resize L240
        ```
        
    + Extract 5 frames per second:
    
        ```sh
        python video2frame.py dataset.json --fps 5
        ```
    
    + Uniformly sample 16 frames per video:
    
        ```sh
        python video2frame.py dataset.json --sample_mode 1 --sample 16
        ```
    
    + Randomly sample 16 frames per video:
    
        ```sh
        python video2frame.py dataset.json --sample_mode 2 --sample 16
        ```
        
    + Use 16 threads to speed-up:
    
        ```sh
        python video2frame.py dataset.json --threads 16
        ```
    
    + Resize the frames to 320x240, extract one frame every two seconds, uniformly sample 32 frames per video, and using 20 threads:
    
        ```sh
        python video2frame.py dataset.json \
            --resize_mode 1 \ 
            --resize 320x240 \
            --fps 0.5 \
            --sample_mode 1 \
            --sample 32 \
            --threads 20
        ```
        
    #### All parameters
    
    ```text
    usage: video2frame.py [-h] [--db_name DB_NAME]
                          [--db_type {LMDB,HDF5,FILE,PKL}] [--tmp_dir TMP_DIR]
                          [--clips CLIPS] [--duration DURATION]
                          [--resize_mode {0,1,2}] [--resize RESIZE] [--fps FPS]
                          [--sample_mode {0,1,2,3}] [--sample SAMPLE]
                          [--threads THREADS] [--keep]
                          annotation_file
    
    positional arguments:
      annotation_file       The annotation file, in json format
    
    optional arguments:
      -h, --help            show this help message and exit
      --db_name DB_NAME     The database to store extracted frames (default: None)
      --db_type {LMDB,HDF5,FILE,PKL}
                            Type of the database (default: HDF5)
      --tmp_dir TMP_DIR     Temporary folder (default: /tmp)
      --clips CLIPS         Num of clips per video (default: 1)
      --duration DURATION   Length of each clip (default: -1)
      --resize_mode {0,1,2}
                            Resize mode
                              0: Do not resize
                              1: 800x600: Resize to WxH
                              2: L600 or S600: keep the aspect ration and scale the longer/shorter side to s (default: 0)
      --resize RESIZE       Parameter of resize mode (default: None)
      --fps FPS             Sample the video at X fps (default: -1)
      --sample_mode {0,1,2,3}
                            Frame sampling options
                              0: Keep all frames
                              1: Uniformly sample n frames
                              2: Randomly sample n continuous frames
                              3: Randomly sample n frames
                              4: Sample 1 frame every n frames (default: 0)
      --sample SAMPLE       How many frames (default: None)
      --threads THREADS     Number of threads (default: 0)
      --keep                Do not delete temporary files at last (default: False)
    ```
    
## Tools

1. `video_folder_to_json.py`

    A json generator where the videos are arranged in this way:

    ```text
    root/swimming/xxx.mp4
    root/swimming/xxy.avi
    root/swimming/xxz.flv
    
    root/dancing/123.mkv
    root/dancing/nsdf3.webm
    root/dancing/asd932_.mov
    ``` 
1. `something_to_json.py`
    
    A json generator that converts the `Something-Something` dataset.

1. `ucf101_to_json.py`

    A json generator that converts the `UCF101` dataset.

## Examples

1. `pytorch_skvideo_dataset.py`

    Get frames using `skvideo` package, when training and evaluating. It is OKay when your batch size is small, and your CPUs are powerful enough.

1. `pytorch_lmdb_video_dataset.py`

    A PyTorch `Dataset` example to read LMDB dataset.

1. `pytorch_hdf5_video_dataset.py`

    A PyTorch `Dataset` example to read HDF5 dataset.
    
    __ALWAYS ENSURE `num_workers=0` OR `num_workers=1` OF YOUR DATA LOADER.__

1. `pytorch_pkl_video_dataset.py`

    A PyTorch `Dataset` example to read pickle dataset.
    
1. `pytorch_file_video_dataset.py`

    A PyTorch `Dataset` example to read image files dataset.