### PART 1: EEG Brain Signal Classification

#### 1. Method
任務目標是要在單一受試者上完成四分類的腦波運動想像（包含左手、右手、腳部動作以及休息狀態）。最大的挑戰是訓練集只有 16 筆資料。為了提取出有效的空間特徵，我們選擇了 *CSP (Common Spatial Pattern)* 演算法。為了解決樣本數不足的問題，我們導入了 *Sliding Window* 來做 Data Augmentation，並搭配帶有 L2 正則化的 *RidgeClassifierCV* 線性分類器與 *Soft-Voting* 決策輔助機制，以此來提升預測的穩定度。

#### 2. Preprocessing Design
*   *Design A (基準對照組：Basic Bandpass + LDA + Hard Voting)：* 
    我們用 Butterworth 帶通濾波器初步取出訊號，經過簡單的滑動切割後，直接丟進沒有任何正則化懲罰的 LDA (Linear Discriminant Analysis) 分類器中。在預測整段 trial 的結果時，我們採用直觀的 **Hard Voting**：也就是看這段 trial 被切出的多個時間窗裡，看哪一個類別是眾數，就當作最終的分類結果。
*   *Design B (最終採用版：Bandpass + Sliding Window Augmentation + Soft-Voting)：* 
    我們鎖定了對運動想像最具鑑別力的 Alpha/Beta 頻段（8-30 Hz）來做濾波。這裡我們刻意加大了資料擴增的力度，把滑動窗口的 Window Size 設為 500、`Step` 設為 50，做出高度重疊的切片。這版最核心的前處理差異在於，我們在 CSP 之後補上了**特徵標準化（StandardScaler）**。而且在最終預測時，我們改採 **Soft-Voting**：不再拿絕對類別去投票，而是把分類器在每個時間窗輸出的 decision_function 保留下來，將所有視窗的分數平均後，再取最高分（`argmax`）當作最終預測。

#### 3. Experimental Results
我們在本地端建構了留一交叉驗證（Leave-One-Out CV），並且透過上傳 Kaggle Public Leaderboard 來驗證模型對未知資料的泛化表現。
*   *Design A：* 表現不佳，Kaggle 上的分數只有 **0.625**。以 16 題測試集來說，大約錯了 6 題。
*   *Design B：* 表現獲得了突破性的提升，Kaggle 分數飆升至 **0.875**，16 題裡面只錯了 2 題。
*註：另外我們也跑過其他頻段（例如 13-30 Hz 或是 70-125 Hz 高頻），但實驗證明 8-30 Hz 萃取出來的特徵最具判別力，效果最好*。*


#### 4. Analysis / Discussion
*Design B 的表現遠勝 Design A，我們認為背後有兩個決定性的原因：*

1.  *強大的抗噪性（Soft-Voting 贏過 Hard Voting 的關鍵）：* 
    腦波訊號（BCI）的特徵出現時間其實很不穩定，受試者可能在 trial 的中後段才開始真正用力想像。如果用 Design A 的 Hard Voting，萬一前段時間都只是靜息期或充滿雜訊，這些雜訊很可能會產生錯誤的極端分類，靠著票數把整個平均結果帶偏。相反地，Design B 的 Soft-Voting 累積的是「決策信心分數」。當模型遇到沒有明顯特徵的雜訊段落時，它的信心分數會在 0 附近游移，不會對總分造成太大影響；但只要掃描到真正顯著的運動想像特徵，分類器就會給出極高分，這等於變相「放大」了真正的特徵訊號，是讓分數躍升的關鍵核心。
2.  *避免過度擬合（Ridge vs LDA）：* 
    由於訓練集只有 16 筆，資料極度高維且稀少。LDA 這種比較「硬」的方法，很容易被少數雜訊騙走，去學出一個過度擬合的決策邊界。Design B 導入的正則化機制成功抑制了這個現象，確保共變異權重不會對生僻的成分過度自信。

#### 5. Task 1 Optional Bonus
*   *改變 Model Family 並引入超參數搜索：* 
    我們捨棄了基本的分類器，換上了 RidgeClassifierCV`。這不只是因為它自帶 L2 Penalty，更重要的是它能自動透過 np.logspace(-3, 3, 10)` 在內部平滑地搜索最佳的懲罰係數，這對極度缺資料的任務來說，是非常有效的正則化手段。
*   *Feature Extraction 後的對齊：* 
    我們觀察到，在經過 CSP 空間轉換提取特徵後，直接丟給分類器效果有時會浮動。因此我們在 Pipeline 中多插了一層 `StandardScaler`。我們發現將這些特徵對齊到平滑基準點上，對於處理容易受雜訊干擾、或是帶有長尾分佈的 EEG 訊號有很大的幫助，能幫助線性模型更容易找出乾淨的切割面。
*   *深度的 Data Augmentation：* 
    雖然滑動窗口算是一種資料處理，但我們把它當作「擴增」的手段，利用 Step=50 這種小步長進行高密度重疊切片，硬是把 16 筆資料擴充成讓模型足以學習穩定權重的資料量。