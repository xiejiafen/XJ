from flask import Flask
from flask_restful import Resource, Api, reqparse
import werkzeug, os

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
        data = parser.parse_args()
        if data['file'] == "":
            return {
                    'data':'',
                    'message':'No file found',
                    'status':'error'
                }
        image = data['file']
        print(image.name)

        if image:
            filename = 'your_image.png'
            image.save(os.path.join(UPLOAD_FOLDER, filename))
            return {
                    'data':'',
                    'message':'img uploaded',
                    'status':'success'
                    }
        return {
                'data':'',
                'message':'Something when wrong',
                'status':'error'
                }


api.add_resource(HelloWorld, '/')
api.add_resource(ImageUpload,'/upload')

if __name__ == '__main__':
    app.run(debug=True)