# Command from here: https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99

TARGET_PATH=$1
FILEID=$2

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O ${TARGET_PATH} && rm -rf /tmp/cookies.txt
