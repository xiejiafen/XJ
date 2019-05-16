from flask import Flask, request
from flask_restful import Resource, Api, reqparse
import werkzeug, os
import traceback
import uuid

app = Flask(__name__, 
            static_url_path='/static', 
            static_folder='static')
api = Api(app)

DIR_NAME = os.path.dirname(__file__)
IMG_FOLDER = 'static/img'
PLUGIN_CONFIG_FILE = 'plugin/path.txt'
parser = reqparse.RequestParser()
parser.add_argument('file',type=werkzeug.datastructures.FileStorage, location='files')

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

class ImageUpload(Resource):
    decorators=[]

    def post(self):
        upload_files = request.files.getlist('file')
        result_files = []

        # 创建目录 
        if not os.path.exists(IMG_FOLDER):
            os.mkdir(IMG_FOLDER)

        try:
            fo = open(PLUGIN_CONFIG_FILE, "w")
            for file in upload_files:

                filepath_arr = file.filename.split('.')
                # filename 需要应使用 secure_filename 但处理后会丢失 _ 前缀
                filepath_arr[-2] = filepath_arr[-2]  + '_' + str(uuid.uuid1())

                filename = '.'.join(filepath_arr)
                
                # 计算输出结果文件名
                filepath_arr[-2] += '_result'
                result_files.append( '/' + IMG_FOLDER + '/' + '.'.join(filepath_arr) )

                # 计算绝对路径存放图片
                upload_path = os.path.join(DIR_NAME, IMG_FOLDER, filename)

                file.save(upload_path)

                # 写入 path.txt
                fo.write( upload_path + '\n' )
            fo.close()
            
            # 执行图像处理程序
            os.system('cd plugin && fast_detect1.exe')

            return {
                'message':'success',
                'payload': result_files
            }
        except:
            return {
                    'message': traceback.format_exc()
                }

api.add_resource(HelloWorld, '/')
api.add_resource(ImageUpload,'/upload')

if __name__ == '__main__':
    app.run(debug=True)