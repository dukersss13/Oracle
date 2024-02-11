import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

url = "https://www.basketball-reference.com/friv/injuries.cgi"

# Fetch the HTML content of the webpage
response = requests.get(url)

injury_report = {"name": [], "team": [], "date": [], "injury_description": []}
# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")
    # Find the table containing the injury data
    injury_table = soup.find("table", {"id": "injuries"})

    if injury_table:
        # Extract rows from the table
        rows = injury_table.find_all("tr")
        # Skip the header row (contains column names)
        header_row = rows[0]
        rows = rows[1:]
        # Iterate through rows and extract data
        for row in rows:
            # Extract data from each column in the row
            columns = row.find_all("td")
            player_name = row.find("th").text.strip()
            team = columns[0].text.strip()
            date = columns[1].text.strip()
            injury_description = columns[2].text.strip()
            injury_report["name"].append(player_name)
            injury_report["team"].append(team)
            injury_report["date"].append(date)
            injury_report["injury_description"].append(injury_description)
    else:
        print("Injury table not found.")
else:
    print("Failed to fetch data. Status code:", response.status_code)


def adjust_datetime_format(date_string: str) -> str:
    # Original date string
    date_string = 'Sat, Feb 3, 2024'

    # Parse the date string into a datetime object
    date_object = datetime.strptime(date_string, '%a, %b %d, %Y')

    # Format the datetime object into MM-DD-YYYY format
    formatted_date = date_object.strftime('%m-%d-%Y')

    return formatted_date

def get_injury_status(injury_report: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and return players who are ruled out
    """
    injury_report["Out"] = ["Out" in description for description in injury_report["injury_description"]]
    injury_report = injury_report[injury_report["Out"]==True]

    return injury_report

injury_report = pd.DataFrame(injury_report)
injury_report["date"] = injury_report["date"].apply(adjust_datetime_format)
injury_report = get_injury_status(injury_report)
