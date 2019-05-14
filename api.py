from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from werkzeug.utils import secure_filename
import werkzeug, os
import traceback

app = Flask(__name__, 
            static_url_path='', 
            static_folder='static')
api = Api(app)
UPLOAD_FOLDER = 'static/upload_img'
parser = reqparse.RequestParser()
parser.add_argument('file',type=werkzeug.datastructures.FileStorage, location='files')

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

class ImageUpload(Resource):
    decorators=[]

    def post(self):
        upload_files = request.files.getlist('file')
        try:
            for file in upload_files:
                filename = secure_filename(file.filename)
                upload_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(upload_path)
            return {
                'message':'success',
            }
        except:
            return {
                    'message': traceback.format_exc()
                }

api.add_resource(HelloWorld, '/')
api.add_resource(ImageUpload,'/upload')

if __name__ == '__main__':
    app.run(debug=True)