import sqlite3
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

class Database:
    def __init__(self, db_path="collected_dataset/plates.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Plates Table
        c.execute('''
            CREATE TABLE IF NOT EXISTS plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                full_image_path TEXT,
                crop_image_path TEXT,
                predicted_text TEXT,
                confidence REAL,
                review_status TEXT DEFAULT 'PENDING',
                batch_id TEXT
            )
        ''')
        
        # Stats Table (Key-Value store)
        c.execute('''
            CREATE TABLE IF NOT EXISTS stats (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def add_entry(self, full_path, crop_path, text, conf, batch_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        ts = datetime.now().isoformat()
        
        c.execute('''
            INSERT INTO plates (timestamp, full_image_path, crop_image_path, predicted_text, confidence, review_status, batch_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (ts, str(full_path), str(crop_path), text, conf, 'PENDING', batch_id))
        
        conn.commit()
        conn.close()
        self.increment_stat('plates_captured')

    def get_all(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM plates ORDER BY timestamp DESC", conn)
        conn.close()
        return df

    def update_status(self, plate_id, status, corrected_text=None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        if corrected_text:
             c.execute("UPDATE plates SET review_status = ?, predicted_text = ? WHERE id = ?", (status, corrected_text, plate_id))
        else:
             c.execute("UPDATE plates SET review_status = ? WHERE id = ?", (status, plate_id))
             
        conn.commit()
        conn.close()

    def get_stat(self, key, default=0):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT value FROM stats WHERE key = ?", (key,))
        res = c.fetchone()
        conn.close()
        return int(res[0]) if res else default

    def increment_stat(self, key):
        val = self.get_stat(key)
        new_val = val + 1
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO stats (key, value) VALUES (?, ?)", (key, str(new_val)))
        conn.commit()
        conn.close()

    def get_pending_count(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM plates WHERE review_status = 'PENDING'")
        res = c.fetchone()[0]
        conn.close()
        return res
