# weather

โปรเจกต์ดึงข้อมูลสภาพอากาศและพยากรณ์ด้วยโมเดล baseline เขียนด้วย Python ล้วน (ใช้แค่ standard library ไม่มี dependency)
ข้อมูลมาจาก [Open-Meteo](https://open-meteo.com/) ซึ่งใช้ฟรีและไม่ต้องมี API key

> ตอนนี้วางไว้ในโฟลเดอร์ `weather/` ของรีโพ `aiml` ไปก่อน โครงสร้างเป็นอิสระในตัว ย้ายไปเป็นรีโพ `weather` แยกได้ทั้งโฟลเดอร์

## โครงสร้าง

```
weather/
├── pyproject.toml
├── src/weather/
│   ├── client.py   # ดึงข้อมูลพยากรณ์ / ข้อมูลย้อนหลังรายชั่วโมงจาก Open-Meteo
│   ├── model.py    # โมเดล baseline (Persistence, AutoRegressive) + ตัววัดผล MAE/RMSE
│   └── cli.py      # คำสั่ง `weather forecast` และ `weather evaluate`
├── tests/          # unit tests (unittest, ไม่เรียกเน็ตจริง)
└── web/
    └── index.html  # เว็บ "ดูฝน Doo Foun" รวมเครื่องมือเช็กฝน น้ำท่วม และเส้นทาง
```

## เว็บ ดูฝน (Doo Foun)

ไฟล์เดียว `web/index.html` เปิดในเบราว์เซอร์ได้เลย ไม่ต้อง build มี 3 แท็บ:

- **หน้าแรก** ลูกโลก เรดาร์ฝน และภาพดาวเทียม (เป็นภาพประกอบ) พร้อมลิงก์ไปดูภาพจริง และขั้นตอนเช็กตามลำดับ
- **Dashboard** ระดับน้ำ การระบายเขื่อน ฝน กทม. และถนนน้ำท่วม (ตอนนี้เป็นข้อมูลตัวอย่าง)
- **เครื่องมือทั้งหมด** ลิงก์เครื่องมือของรัฐและเอกชน กรองตามหมวดและค้นหาได้

สายด่วนอยู่ท้ายทุกหน้า ขั้นต่อไปคือให้สคริปต์ Python ในโปรเจกต์นี้ดึงข้อมูลจริงมาใส่ Dashboard

## การออกแบบ

1. **ชั้นข้อมูล (`client.py`)** – แปลง response JSON เป็น `HourlySeries` (เวลา + คอลัมน์ต่อตัวแปร) และตรวจว่าความยาวทุกคอลัมน์ตรงกัน
2. **ชั้นโมเดล (`model.py`)** – ทุกโมเดลมี `fit(series)` / `predict(history, horizon)` เหมือนกัน จึงสลับไปใช้โมเดล ML ที่ซับซ้อนกว่า (เช่น gradient boosting, LSTM) ได้โดยไม่ต้องแก้ส่วนอื่น
   - `Persistence` – ใช้ค่าของ 24 ชั่วโมงก่อนหน้าเป็นคำตอบ เป็นเกณฑ์ขั้นต่ำที่โมเดลอื่นต้องชนะให้ได้
   - `AutoRegressive` – linear regression บนค่า 24 ชั่วโมงล่าสุด (ridge + least squares)
3. **การประเมินผล** – `evaluate()` ใช้ข้อมูลก่อนหน้าเทรน แล้วทดสอบกับ `horizon` ชั่วโมงสุดท้าย

## การใช้งาน

```bash
cd weather
pip install -e .                       # หรือใช้ PYTHONPATH=src แทนการติดตั้ง

weather forecast --hours 12            # พยากรณ์อุณหภูมิ (ค่าเริ่มต้น: กรุงเทพฯ)
weather --lat 18.79 --lon 98.98 forecast   # เชียงใหม่
weather evaluate --days 60 --horizon 24    # เทียบโมเดล baseline กับข้อมูลย้อนหลัง
```

## ทดสอบ

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## ขั้นต่อไปที่เสนอ

- เพิ่มฟีเจอร์ (ความชื้น ลม ชั่วโมงของวัน ฤดูกาล) แล้วลองใช้ scikit-learn / LightGBM
- ทำ backtest แบบ rolling-origin หลายหน้าต่างเวลา แทนการแบ่ง train/test ครั้งเดียว
- เก็บข้อมูลย้อนหลังลงไฟล์ (`data/`) เพื่อไม่ต้องดึงซ้ำ
- ทำหน้า dashboard หรือ API สำหรับแสดงผล
