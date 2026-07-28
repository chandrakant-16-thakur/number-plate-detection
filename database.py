import sqlite3

conn = sqlite3.connect("plates.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS plates(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT,
    image_name TEXT,
    date_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("Database Created Successfully!")