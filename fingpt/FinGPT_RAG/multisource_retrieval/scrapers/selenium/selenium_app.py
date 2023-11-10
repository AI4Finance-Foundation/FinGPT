from selenium import webdriver
from selenium.webdriver.chromium.service import ChromiumService
from webdriver_manager.chrome import ChromeDriverManager

# Set up ChromeOptions
options = webdriver.ChromeOptions()
# options.binary_location = "/Users/tianyu/Desktop/Coding/Network/chrome/chrome-mac-arm64"

# Start Chrome using a specific ChromeDriver
executable_path='/Users/tianyu/Desktop/Coding/Network/chrome/chromedriver-mac-arm64'
executable_path=ChromeDriverManager().install()
service=ChromiumService(executable_path=executable_path)
driver = webdriver.Chrome(service=service, options=options)

driver.get('https://www.google.com')
print(driver.title)
driver.quit()
