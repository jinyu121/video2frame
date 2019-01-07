# video2frame

Video2frame is also an easy-to-use tool to extract frames from video.

## Why this tool

[Forwchen's vid2frame tool](https://github.com/forwchen/vid2frame) is great, but I am always confused by their parameters. At the same time, I also want to add something I need to the tool. 

So I re-wrote the code. And now, it is a new wheel. It is hard to make a PR since I changed the code style. 

## How to use

1. ### Establish the environment
    
    We recommend to use [conda](https://conda.io/) to establish the environment. Just using 
    
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
    [
        {
            "path":"path/to/the/video/file_1.mp4"
        },
        {
            "path":"path/to/the/video/file_2.mp4"
        }
    ]
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
        python video2frame.py dataset.json -t 16
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
    usage: video2frame.py [-h] [--db_name DB_NAME] [--db_type {LMDB,HDF5}]
                          [--tmp_dir TMP_DIR] [--resize_mode {0,1,2}]
                          [--resize RESIZE] [--fps FPS] [--sample_mode {0,1,2,3}]
                          [--sample SAMPLE] [-t THREADS] [-nrm]
                          annotation_file
    
    positional arguments:
      annotation_file       The annotation file, in json format
    
    optional arguments:
      -h, --help            show this help message and exit
      --db_name DB_NAME     The database to store extracted frames
      --db_type {LMDB,HDF5} Type of the database, LMDB or HDF5
      --tmp_dir TMP_DIR     Tmp dir
      --resize_mode {0,1,2} Resize mode
                              0: Do not resize
                              1: 800x600: Resize to W*H
                              2: L600 or S600: keep the aspect ration and scale the longer/shorter side to s
      --resize RESIZE       Parameter of resize mode
      --fps FPS             Sample the video at X fps
      --sample_mode {0,1,2,3}
                            Frame sampling options
                              0: Keep all frames
                              1: Uniformly sample n frames
                              2: Randomly sample n frames
                              3: Mod mode
      --sample SAMPLE       Parameter of sample mode
      -t THREADS, --threads THREADS
                            Number of threads
      -nrm, --not_remove    Do not delete tmp files at last
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
    
    A json generator that converts the Something-Something dataset.
    
## Examples

1. `pytorch_hdf5_video_dataset.py`

    A PyTorch `Dataset` example, as well as an HDF5 database checker.
    
    The json file should like:
    
    ```json
    [
        {
            "path":"path/to/the/video/file_1.mp4",
            "class": 1
        },
        {
            "path":"path/to/the/video/file_2.mp4",
            "class": 2
        }
    ]
    ```