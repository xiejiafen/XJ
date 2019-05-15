需要新建 ./plugin 目录将 ./de 项目产出物 copy 在其内

## 运行接口
python api.py

## 测试页面
http://127.0.0.1:5000/static/demo.html

## 接口地址
http://127.0.0.1:5000/upload

## request FormData
多文件上传

## response json
```
{
    message: 'success',
    payload: [
        "/static/img/_1_result.jpg"
    ]
}
```