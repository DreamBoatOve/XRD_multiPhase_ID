import requests
from bs4 import BeautifulSoup
import time
import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np


"""
Author
	Sunny Yang
"""

class RRUFFScraper:
    def __init__(self):
        self.base_url = "https://rruff.info"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_mineral_links(self, letter):
        """获取某个字母开头的所有矿物链接"""
        url = f"{self.base_url}/index.php/r=lookup_minerals/letter={letter}/calling_form=frm_sample_search/name_field=txt_mineral/id_field=(letter)"
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        links = []
        for strong_tag in soup.find_all('strong'):
            a_tag = strong_tag.find('a')
            if a_tag and 'SubmitWin' in a_tag.get('href', ''):
                href = a_tag['href']
                mineral_name = href.split("'")[1]
                mineral_id = href.split("'")[3]
                links.append((mineral_name, mineral_id))
        return links

    def download_xray_data(self, mineral_info, output_dir='rruff_files'):
        """下载 X-ray Data (XY - Processed) 和 X-ray Data (XY - RAW) 文件"""
        mineral_name, mineral_id = mineral_info
        url = f"{self.base_url}/{mineral_name}/R{mineral_id}"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            powder_table = None
            for table in soup.find_all('table'):
                th = table.find('th')
                if th and "POWDER DIFFRACTION" in th.text:
                    powder_table = table
                    break

            if powder_table:
                download_links = {}

                for tr in powder_table.find_all('tr'):
                    for a in tr.find_all('a', href=True):
                        if "X-ray Data (XY - Processed)" in a.text or "X-ray Data (XY - RAW)" in a.text:
                            link = a['href']
                            if link.startswith('http'):
                                xray_data_link = link
                            else:
                                xray_data_link = self.base_url + link

                            file_info = a.text.strip().replace(" ", "_").replace("(", "").replace(")", "").lower()
                            download_links[xray_data_link] = file_info

                if not download_links:
                    return False

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                for xray_data_link, file_info in download_links.items():
                    filename = f"{output_dir}/{mineral_name}_{file_info}.txt"

                    file_response = requests.get(xray_data_link, headers=self.headers)
                    with open(filename, 'wb') as f:
                        f.write(file_response.content)

                return True
            else:
                return False

        except requests.exceptions.RequestException as e:
            print(f"Error during requests to {url}: {str(e)}")
            print(soup.prettify())
            return False
        except Exception as e:
            print(f"Error processing {mineral_info}: {str(e)}")
            return False

    def scrape_all(self, start_letter='a', end_letter='z', delay=2):
        """下载从start_letter到end_letter的所有RRUFF文件"""
        success_count = 0

        for letter in range(ord(start_letter.lower()), ord(end_letter.lower()) + 1):
            letter = chr(letter)

            mineral_links = self.get_mineral_links(letter)

            # 使用 tqdm 创建进度条
            with tqdm(total=len(mineral_links), desc=f"Processing letter {letter}") as pbar:
                for mineral_info in mineral_links:
                    if self.download_xray_data(mineral_info):
                        success_count += 1
                    time.sleep(delay)
                    pbar.update(1)  # 更新进度条

        print(f"\nDownload completed! Successfully downloaded {success_count} X-ray Data files.")


class XRDDataset(Dataset):
    def __init__(self, directory, max_length=None):
        """
        Args:
            directory: Path to the directory containing the RRUFF files.
            max_length (int, optional): Maximum length to pad/truncate XRD data.
                If None, uses the longest sequence in the dataset.
        """
        self.directory = directory
        self.filepaths = []
        self.max_length = max_length

        # Collect filepaths and determine max_length if not provided
        temp_max_length = 0
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.endswith('.txt'):
                self.filepaths.append(entry.path)
                if self.max_length is None:
                    try:
                        xrd_data = self._load_xrd_data(entry.path)
                        temp_max_length = max(temp_max_length, len(xrd_data))
                    except (ValueError, FileNotFoundError) as e:
                        print(f"Skipping file: {e}")
                        self.filepaths.pop()

        if self.max_length is None:
            self.max_length = temp_max_length
        print(f"max_length: {self.max_length}")


    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        xrd_data, name, crystal_system, file_type = self._load_data(filepath)

        xrd_data = self._pad_or_truncate(xrd_data, self.max_length)

        sample = {
            'xrd_data': xrd_data,
            'name': name,
            'crystal_system': crystal_system,
            'file_type': file_type  # Add file type
        }
        return sample


    def _load_data(self, filepath):
        """Loads data, name, crystal system, and file type from a single file."""
        with open(filepath, 'r') as f:
            lines = f.readlines()

        name = ""
        crystal_system = ""
        xrd_data_start = -1

        # Extract file type from filename
        filename = os.path.basename(filepath)
        if "_-_raw" in filename:
            file_type = "raw"
        elif "_-_processed" in filename:
            file_type = "processed"
        else:
            file_type = "unknown"

        for i, line in enumerate(lines):
            if line.startswith("##NAMES="):
                name = line[8:].strip()
            elif line.startswith("##CELL PARAMETERS="):
                match = re.search(r"crystal system:\s*(\w+)", line)
                if match:
                    crystal_system = match.group(1)
            # MODIFIED REGEX HERE:  Allow optional whitespace
            elif re.match(r"^\d+\.\d+,\s*\d+\.\d+$", line.strip()) or re.match(r"^\d+,\s*\d+$", line.strip()):
                xrd_data_start = i
                break

        if xrd_data_start == -1:
            print(f"No XRD data found in: {filepath}")
            return np.empty((0, 2), dtype=np.float32), "", "", ""

        xrd_lines = lines[xrd_data_start:]
        xrd_data = []
        for line in xrd_lines:
            if line.startswith("##END="):
                break
            try:
                # Handle both float and int pairs
                parts = line.strip().split(",")
                angle, intensity = map(float, parts)  # Convert to float
                xrd_data.append([angle, intensity])
            except ValueError:
                print(f"Warning: Skipping malformed line in {filepath}: {line.strip()}")
                continue

        return np.array(xrd_data, dtype=np.float32), name, crystal_system, file_type


    def _load_xrd_data(self, filepath):
        """Helper function for max_length calculation (loads only XRD data)."""
        with open(filepath, 'r') as f:
             lines = f.readlines()

        xrd_data_start = -1
        for i, line in enumerate(lines):
             # MODIFIED REGEX HERE:  Allow optional whitespace, and integer pairs.
            if re.match(r"^\d+\.\d+,\s*\d+\.\d+$", line.strip()) or re.match(r"^\d+,\s*\d+$", line.strip()):
                xrd_data_start = i
                break

        if xrd_data_start == -1:
          return []

        xrd_lines = lines[xrd_data_start:]
        xrd_data = []
        for line in xrd_lines:
          if line.startswith("##END="):
            break
          try:
              # Handle both float and int pairs
              parts = line.strip().split(",")
              angle, intensity = map(float, parts) # Convert to float
              xrd_data.append([angle, intensity])
          except ValueError:
              print(f"Skipping malformed line (helper) in {filepath}: {line}")
              continue
        return xrd_data

    def _pad_or_truncate(self, xrd_data, max_length):
        """Pads or truncates the XRD data."""
        if len(xrd_data) > max_length:
            xrd_data = xrd_data[:max_length]
        elif len(xrd_data) < max_length:
            padding = np.zeros((max_length - len(xrd_data), 2), dtype=np.float32)
            xrd_data = np.vstack((xrd_data, padding))
        return torch.tensor(xrd_data)



# --- Example Usage ---
if __name__ == '__main__':
    test_dir = './rruff_files'  # Real directory
    dataset = XRDDataset(directory=test_dir, max_length=10000)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        print("  XRD Data Batch Shape:", batch['xrd_data'].shape)
        print(f"  Name Type: {type(batch['name'])}")
        print(f"  Crystal System Type: {type(batch['crystal_system'])}")
        print(f"  File Type: {type(batch['file_type'])}")

        for j in range(batch['xrd_data'].shape[0]):
            print(f"    Sample {j} in Batch {i}:")
            print(f"      Crystal System: {batch['crystal_system'][j]}")
            print(f"      Name: {batch['name'][j]}")
            print(f"      File Type: {batch['file_type'][j]}")

        if i == 20:
            break

if __name__ == "__main__":
    scraper = RRUFFScraper()
    scraper.scrape_all('a', 'b',0.01)