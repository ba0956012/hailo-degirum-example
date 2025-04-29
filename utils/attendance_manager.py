import sqlite3
from datetime import datetime

class AttendanceManager:
    def __init__(self, db_path='attendance.db'):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """建立資料表（如果尚未存在）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            date DATE NOT NULL,
            first_punch DATETIME NOT NULL,
            last_punch DATETIME NOT NULL,
            UNIQUE(employee_id, date)
        )
        ''')

        conn.commit()
        conn.close()

    def punch(self, employee_id):
        """打卡（如果今天有紀錄就更新 last_punch，沒有就新增）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        timestamp_str = now.strftime('%Y-%m-%d %H:%M:%S')

        try:
            cursor.execute('''
            INSERT INTO attendance (employee_id, date, first_punch, last_punch)
            VALUES (?, ?, ?, ?)
            ''', (employee_id, date_str, timestamp_str, timestamp_str))
            print(f"[打卡] 員工 {employee_id} 第一次打卡 {timestamp_str}")

        except sqlite3.IntegrityError:
            cursor.execute('''
            UPDATE attendance
            SET last_punch = ?
            WHERE employee_id = ? AND date = ?
            ''', (timestamp_str, employee_id, date_str))
            print(f"[打卡] 員工 {employee_id} 更新最後打卡時間 {timestamp_str}")

        conn.commit()
        conn.close()

    def get_summary(self, date_str=None):
        """查詢某天所有員工的打卡情形（不給日期就是今天）"""
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT employee_id, first_punch, last_punch
        FROM attendance
        WHERE date = ?
        ''', (date_str,))

        rows = cursor.fetchall()

        print(f"--- {date_str} 打卡紀錄 ---")
        for row in rows:
            employee_id, first_punch, last_punch = row
            print(f"員工 {employee_id} | 最早打卡: {first_punch} | 最晚打卡: {last_punch}")

        conn.close()

    def delete_all(self):
        """清空所有打卡紀錄（小心使用）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM attendance')

        conn.commit()
        conn.close()
        print("已清空所有打卡紀錄。")

