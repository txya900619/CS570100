# Data Science HW1

## 目標

給定 `transactions` 和 `min support` (頻率)，實作演算法找出 frequent patterns  
可使用 Python3 或 C++，演算法不限 (Apriori、FP Growth 等皆可)  
不得使用 frequent patterns 相關的 library

## 輸入

- 輸入檔為存有 transactions 的 txt 檔  
- Item 以數字表示，範圍為 0 ~ 999  
- Transactions 最多 100,000 筆  
- 每筆 transaction 最多 200 個 item  
- 每一行代表一筆 transaction，筆與筆之間以逗號 `,` 區隔，無空格  
- 換行採用 `\n` (LF)，而非 `\r\n` (CRLF)
- 範例: `sample.txt`

## 輸出

- 輸出一個 txt 檔  
- 每一行為一組 frequent pattern，後面接上 `:` 再接上 support (出現頻率)  
  - **Example:** `1,2,3:0.2500` (Support 四捨五入到小數點後第 4 位)
- 輸出的部分不需要特別排序，助教評分時會自行處理
- 範例:  `sample.txt (min support = 0.2)` 計算出的 output 為 `sample_out.txt`

## 實作方式

- gemini 2.5 pro 生成
- [過程](https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221Ll-Rmltctdnwk7al7w7Ltaw7TZd1gH7d%22%5D,%22action%22:%22open%22,%22userId%22:%22113835453565800784030%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing)