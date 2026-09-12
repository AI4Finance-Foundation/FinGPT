import os
import pandas as pd
import numpy as np
import requests
from lxml import etree
import multiprocessing as mp
import json
import signal
import sys

# The result_path should be the results with only titles which is the IN path
result_path = r"D:\python_project\FinRL-Meta\experiment\scrape\results"

# The result_with_content_path should be the results with titles and contents which is the OUT path
result_with_content_path = r"D:\python_project\FinRL-Meta\experiment\scrape\results_with_content"
link_base = "https://guba.eastmoney.com"


def get_one_content(x):
    url = link_base + x["content link"]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/112.0",
        "Referer": "https://guba.eastmoney.com/",
    }
    
    # Optional proxy configuration - set to None if not needed
    tunnel = os.environ.get("KUAIDAILI_TUNNEL", None)
    username = os.environ.get("KUAIDAILI_USERNAME", None)
    password = os.environ.get("KUAIDAILI_PASSWORD", None)
    
    proxies = None
    if tunnel and username and password:
        proxies = {
            "http": "http://%(user)s:%(pwd)s@%(proxy)s/" % {"user": username, "pwd": password, "proxy": tunnel},
            "https": "http://%(user)s:%(pwd)s@%(proxy)s/" % {"user": username, "pwd": password, "proxy": tunnel}
        }
    
    requests.DEFAULT_RETRIES = 5  # more retrys
    s = requests.session()
    s.keep_alive = False  # close connection when finished
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.get(url = url, headers = headers, proxies= proxies, timeout=15)
            if response.status_code == 200:
                res = etree.HTML(response.text)
                script_texts = res.xpath("//script//text()")
                if len(script_texts) >= 2:
                    res = script_texts[1]  # Get the second script
                    json_data = json.loads(res[17:])

                    if isinstance(json_data, dict):
                        res_series = pd.Series(json_data)
                    else:
                        res_series = pd.Series()

                    return res_series
                else:
                    print(f"Warning: Could not find script data for {url}")
                    return pd.Series()
            else:
                print(f"Warning: Got status code {response.status_code} for {url}")
                retry_count += 1
        except requests.exceptions.Timeout:
            print(f"Timeout error for {url}, retry {retry_count + 1}/{max_retries}")
            retry_count += 1
        except requests.exceptions.RequestException as e:
            print(f"Request error for {url}: {e}, retry {retry_count + 1}/{max_retries}")
            retry_count += 1
        except Exception as e:
            print(f"Unexpected error for {url}: {e}")
            return pd.Series()
    
    print(f"Failed to fetch content for {url} after {max_retries} retries")
    return pd.Series()


def get_content(file_name):
    try:
        df = pd.read_csv(os.path.join(result_path, file_name))
        print(f"Processing file: {file_name} with {len(df)} rows")

        new_columns = ['post_user', 'post_guba', 'post_publish_time', 'post_last_time',
           'post_display_time', 'post_ip', 'post_checkState', 'post_click_count',
           'post_forward_count', 'post_comment_count', 'post_comment_authority',
           'post_like_count', 'post_is_like', 'post_is_collected', 'post_type',
           'post_source_id', 'post_top_status', 'post_status', 'post_from',
           'post_from_num', 'post_pdf_url', 'post_has_pic',
           'has_pic_not_include_content', 'post_pic_url', 'source_post_id',
           'source_post_state', 'source_post_user_id', 'source_post_user_nickname',
           'source_post_user_type', 'source_post_user_is_majia',
           'source_post_pic_url', 'source_post_title', 'source_post_content',
           'source_post_abstract', 'source_post_ip', 'source_post_type',
           'source_post_guba', 'post_video_url', 'source_post_video_url',
           'source_post_source_id', 'code_name', 'product_type', 'v_user_code',
           'source_click_count', 'source_comment_count', 'source_forward_count',
           'source_publish_time', 'source_user_is_majia', 'ask_chairman_state',
           'selected_post_code', 'selected_post_name', 'selected_relate_guba',
           'ask_question', 'ask_answer', 'qa', 'fp_code', 'codepost_count',
           'extend', 'post_pic_url2', 'source_post_pic_url2', 'relate_topic',
           'source_extend', 'digest_type', 'source_post_atuser',
           'post_inshare_count', 'repost_state', 'post_atuser', 'reptile_state',
           'post_add_list', 'extend_version', 'post_add_time', 'post_modules',
           'post_speccolumn', 'post_ip_address', 'source_post_ip_address',
           'post_mod_time', 'post_mod_count', 'allow_likes_state',
           'system_comment_authority', 'limit_reply_user_auth', 'post_id',
           'post_title', 'post_content', 'post_abstract', 'post_state']
        
        # Process with progress tracking
        print(f"Starting content extraction for {file_name}...")
        df[new_columns] = df.apply(lambda x:get_one_content(x), axis = 1, result_type= "expand", )
        
        to_path = os.path.join(result_with_content_path, file_name)
        df.to_csv(to_path, index = False)
        print(f"Successfully saved {file_name} to {to_path}")
        
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        # Try to save what we have so far
        try:
            to_path = os.path.join(result_with_content_path, file_name)
            if 'df' in locals():
                df.to_csv(to_path, index = False)
                print(f"Saved partial results for {file_name}")
        except Exception as save_error:
            print(f"Could not save partial results: {save_error}")

if __name__ == "__main__":
    pool_list = []
    res_list = []
    
    # Reduce number of processes to avoid overwhelming the system
    num_processes = min(4, mp.cpu_count())
    print(f"Using {num_processes} processes for parallel processing")
    
    pool = mp.Pool(processes = num_processes)
    
    file_list = os.listdir(result_path)
    print(f"Found {len(file_list)} files to process")
    
    # Process files one at a time if there are issues with multiprocessing
    if len(file_list) < 3:
        print("Processing files sequentially due to small file count")
        for i in file_list:
            print(i)
            get_content(i)
    else:
        # Use multiprocessing for larger file counts
        for i in file_list:
            print(i)
            res = pool.apply_async(get_content, args = (i,), error_callback = lambda x:print(f"Error in async processing: {x}"))
            pool_list.append(res)

        pool.close()
        pool.join()

        # # 获取运行结果
        # for i in pool_list:
        #     res_list.append(i.get())

    print("All Done!")
