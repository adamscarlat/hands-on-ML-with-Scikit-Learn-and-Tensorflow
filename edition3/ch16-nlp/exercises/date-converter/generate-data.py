import random
import csv

MONTH_MAP = {
  "January": "01",
  "February": "02",
  "March": "03",
  "April": "04",
  "May": "05",
  "June": "06",
  "July": "07",
  "August": "08",
  "September": "09",
  "October": "10",
  "November": "11",
  "December": "12",
}

def generate_dates(num):
  num_per_month = num // 12
  dates_dataset = []
  for month_name, month_num in MONTH_MAP.items():
    for _ in range(num_per_month):
      year_num = random.randint(1000, 9999)
      day_num = random.randint(1,31)

      # Add a prefix zero before day num below 10
      day_str = str(day_num)
      if day_num < 10:
        day_str = f"0{day_str}"
      
      date_train = f"{month_name} {day_num}, {year_num}"
      date_label = f"{year_num}-{month_num}-{day_str}"

      dates_dataset.append((date_train, date_label))
  
  return dates_dataset

dates_dataset = generate_dates(10000)
with open("ch16-nlp/exercises/date-converter/data.csv", "w", newline='') as f:
  csv_writer = csv.writer(f)
  csv_writer.writerow(['date_from', 'date_to'])
  csv_writer.writerows(dates_dataset)

