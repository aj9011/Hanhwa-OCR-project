# Hanhwa-OCR-project
PSE NET + TPS-ResNet + Connection Algorithm

구동 방식
현재 Demo용으로 만든 코드라 미리 레이블링이 완료된 이미지 데이터가 아니면 작동하지 않습니다.
AI모델의 결과물을 활용하여 연결을 하지 않고, 미리 레이블링 된 GT로 연결을 진행합니다.


templates/index.html에서 메인 View


Dropzone에 이미지 업로드 하면 서버로 이미지 전송


uploads/uuid(사용자 개별 식별 아이디)/original에 이미지 저장


서버 내에서 assets폴더 안에 있는 스크립트들 순차적으로 실행


inference.py안에서 inferencepse.py->croppse.py->inferenceocr.py->tableunderstanding->visualizer.py 순


최종 결과물 {이미지이름 : DataFrame} 반환


이미지 데이터 base64로 인코딩 후 클라이언트로 전송


templates/result.html로 결과 전송




구동 필요 파일들


assets/save_model/* : 학습된 모델 파일들 필요구글드라이브 링크


assets/demo/* : GT데이터들 필요구글드라이브 링크




가상환경 세팅 방법
Anaconda의 가상환경 사용을 추천
conda create -n ## python=3.5
pip install opencv-python PyYaml natsort Pillow==6.1 tensorflow-gpu==1.13.1 flask pandas easydict matplotlib scipy lmdb
