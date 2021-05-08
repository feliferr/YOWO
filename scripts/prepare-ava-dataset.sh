
# 1- Downloading AVA dataset

echo " ====== 1- Downloading AVA dataset ====="

DATA_DIR="datasets/AVA/videos"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

wget https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt

for line in $(cat ava_file_names_trainval_v2.1.txt)
do
  FILE=${DATA_DIR}/${line}
  if test -f "$FILE"; then
    echo "$FILE alredy exists!"
  else
    echo "Downloading file and save it in $FILE"
    wget https://s3.amazonaws.com/ava-dataset/trainval/$line -P ${DATA_DIR}
  fi
done
echo "Done!"

# 2 - Cut each video from its 15th to 30th minute

echo " ====== 2 - Cut each video from its 15th to 30th minute ====="

IN_DATA_DIR="datasets/AVA/videos"
OUT_DATA_DIR="datasets/AVA/videos_15min"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  out_name="${OUT_DATA_DIR}/${video##*/}"
  if [ ! -f "${out_name}" ]; then
    ffmpeg -ss 900 -t 901 -i "${video}" "${out_name}"
  fi
done
echo "Done!"

# 3 - Extract frames

echo " ====== 3 - Extract frames ====="

IN_DATA_DIR="datasets/AVA/videos_15min"
OUT_DATA_DIR="datasets/AVA/frames"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  video_name=${video##*/}

  if [[ $video_name = *".webm" ]]; then
    video_name=${video_name::-5}
  else
    video_name=${video_name::-4}
  fi

  out_video_dir=${OUT_DATA_DIR}/${video_name}

  if [[ ! -d "${out_video_dir}" ]]; then
    echo "${out_video_dir} doesn't exist. Creating it.";
    mkdir -p "${out_video_dir}"

    out_name="${out_video_dir}/${video_name}_%06d.jpg"

    ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
  fi
done



# 4 - Download annotations

echo " ====== 4 - Download annotations ====="

DATA_DIR="datasets/AVA/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

wget https://research.google.com/ava/download/ava_train_v2.1.csv -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_val_v2.1.csv -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_action_list_v2.1_for_activitynet_2018.pbtxt -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_train_excluded_timestamps_v2.1.csv -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_val_excluded_timestamps_v2.1.csv -P ${DATA_DIR}


# 5 - Download "frame lists" (train, val) and put them in the "frame_lists" folder

echo " ====== 5 - Download frame lists (train, val) and put them in the frame_lists folder ====="

DATA_DIR="datasets/AVA/frame_lists"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

wget https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/frame_lists/train.csv -P ${DATA_DIR}
wget https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/frame_lists/val.csv -P ${DATA_DIR}


# 6 -  Download person boxes (train, val, test) and put them in the annotations folder

echo " ====== 6 -  Download person boxes (train, val, test) and put them in the annotations folder ====="

DATA_DIR="datasets/AVA/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

wget https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_train_predicted_boxes.csv -P ${DATA_DIR}
wget https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_val_predicted_boxes.csv -P ${DATA_DIR}
wget https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_test_predicted_boxes.csv -P ${DATA_DIR}