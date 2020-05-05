from flask import Flask, redirect, render_template, request, session, url_for
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_reverse_proxy_fix.middleware import ReverseProxyPrefixFix
from uuid import uuid4
from datetime import datetime
import inference
import os

import base64

app = Flask(__name__)

app.config['REVERSE_PROXY_PATH'] = '/nakedreader'
app.config['SECRET_KEY'] = 'agilesoda'

# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

Dropzone(app)
ReverseProxyPrefixFix(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    
    # set session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
    if "uuid" not in session:
        session['uuid'] = "_".join([datetime.now().strftime("%y%m%d_%H%M%S"), uuid4().hex])
    
    # list to hold our uploaded image urls
    file_urls = session['file_urls']
    uuid = session['uuid']

    # handle image upload from Dropszone
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            
            # save the file with to our photos folder
            filename = photos.save(
                file,
                folder=uuid + "/original",
                name=file.filename
            )
            try:
                base_dir = os.getcwd() + '/uploads/' + uuid + "/" 
                os.makedirs(base_dir + 'pse')
                os.makedirs(base_dir + 'ocr')
                os.makedirs(base_dir + 'txt')
                os.makedirs(base_dir + 'result')
            except OSError as e:
                if e.errno != os.errno.EEXIST:
                    raise            
            # append image urls
            file_urls.append(file.filename)
        session['file_urls'] = file_urls
        return "uploading..."

    # return dropzone template on GET request
    return render_template('index.html')


@app.route('/results')
def results():
    
    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('index'))
    images = []
    tables = []
    try:
        file_urls = session['file_urls']
        session.pop('file_urls', None)
        uuid = session['uuid']
        session.pop('uuid', None)
        file_path = "./uploads/" + uuid + "/original/"
        output = inference.inference(file_path, file_urls)
        
        for file_name in file_urls:
            img_name = output[file_name]['img']
            with open(img_name, 'rb') as img_file:
                images.append(base64.b64encode(img_file.read()))
            tables.append(output[file_name]['df'])
            
    except Exception as identifier:
        print(identifier)
        pass
    
    try :
        folder_list = glob("./uploads/" + uuid + "/*")
        folder_list.sort(key=os.path.getmtime)
        _ = [shutil.rmtree(x, ignore_errors=True) for x in folder_list[:-20]]
        
    except :
        pass
    
    return render_template('results.html', counts=len(images), images=images, tables=tables)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)