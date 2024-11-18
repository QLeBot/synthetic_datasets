from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import requests
import os
from datetime import datetime
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver 
from selenium.webdriver.chrome.service import Service as ChromeService

def download_single_face(thread_id, output_queue):
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    # Initialize the driver
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
    
    try:
        # Navigate to the website
        url = "https://thispersondoesnotexist.com/"
        driver.get(url)
        
        # Wait for the image to load
        time.sleep(5)
        
        # Find the image element and download
        img_element = driver.find_element('xpath', '//img')
        img_url = img_element.get_attribute('src')
        img_data = requests.get(img_url).content
        
        # Create 'ai_faces' directory if it doesn't exist
        if not os.path.exists('ai_faces'):
            os.makedirs('ai_faces')
        
        # Save the image with timestamp and thread ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'ai_faces/ai_face_{timestamp}_thread{thread_id}.jpg'
        
        with open(filename, 'wb') as handler:
            handler.write(img_data)
            
        output_queue.put((thread_id, filename))
        
    except Exception as e:
        print(f"Thread {thread_id}: An error occurred: {str(e)}")
        output_queue.put((thread_id, None))
        
    finally:
        driver.quit()

def download_ai_face(num_images=1, max_threads=1):
    output_queue = Queue()
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit tasks to the thread pool
        futures = [executor.submit(download_single_face, i, output_queue) 
                  for i in range(num_images)]
    
    # Collect and print results
    successful_downloads = 0
    for _ in range(num_images):
        thread_id, filename = output_queue.get()
        if filename:
            print(f"Thread {thread_id}: Downloaded successfully: {filename}")
            successful_downloads += 1
    
    print(f"\nDownload complete. Successfully downloaded {successful_downloads}/{num_images} images.")

# Example usage
if __name__ == "__main__":
    # Download 5 images
    download_ai_face(100, max_threads=1)

