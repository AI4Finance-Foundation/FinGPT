package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/sqweek/dialog"
)

type Result struct {
	Number  int    `json:"number"`
	Success bool   `json:"success"`
	Content string `json:"content"`
}

func main() {
	startNumber := 3490000
	endNumber := 4508859

	results := make(chan Result)
	var wg sync.WaitGroup

	folder, _ := dialog.Directory().Title("Select Save Folder").Browse()

	for i := startNumber; i <= endNumber; i++ {
		wg.Add(1)
		go func(number int) {
			defer wg.Done()

			url := fmt.Sprintf("https://seekingalpha.com/api/v3/articles/%d", number)
			resp, err := http.Get(url)
			if err != nil {
				results <- Result{Number: number, Success: false, Content: ""}
				return
			}
			defer resp.Body.Close()

			body, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				results <- Result{Number: number, Success: false, Content: ""}
				return
			}

			var data struct {
				Attributes struct {
					Content string `json:"content"`
				} `json:"data"`
			}
			err = json.Unmarshal(body, &data)
			if err != nil {
				results <- Result{Number: number, Success: false, Content: ""}
				return
			}


			results <- Result{Number: number, Success: true, Content: data.Attributes.Content}
		}(i)

		if i%100 == 0 {
			go func() {
				wg.Wait()
				close(results)
			}()

			saveResultsToFile(folder, results)
			wg = sync.WaitGroup{}
		}
	}

	saveResultsToFile(folder, results)
}

func saveResultsToFile(folder string, results <-chan Result) {
	filePath := filepath.Join(folder, "results.json")
	file, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Printf("Failed to open file: %v\n", err)
		return
	}
	defer file.Close()

	encoder := json.NewEncoder(file)

	for result := range results {
		err := encoder.Encode(result)
		if err != nil {
			fmt.Printf("Failed to encode result: %v\n", err)
		}
	}
}
