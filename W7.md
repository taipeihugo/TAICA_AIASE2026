# AIASE 2026 W7 — Agentic Stack: Inter- and Intra-Connectivity

**課程**：生成式 AI 應用系統與工程 (Generative AI Application Systems and Engineering)  
**授課教師**：莊坤達 Kun-Ta Chuang  
**單位**：國立成功大學 資訊工程學系 (NCKU CSIE)

---

## 第一部分：Agentic Workflow 入門——從基礎套件理解運作機制

> 以 autoresearch 與 gstack 為例，談「知其所以然」的代理人工程學習路徑

### 1.1 為什麼要從具「基礎」流程的應用看起？

- 市面已有許多成熟整合框架（OpenClaw、LangGraph、AutoGen、CrewAI），封裝了多代理人協作、訊息路由、狀態管理與工具調用
- **危險**：你看到的只是「運作的結果」，而不是「運作的原理」
  - 出錯時不知道為什麼錯；客製化時不知道從哪裡改
  - 框架原始碼高度抽象化，讓你永遠只是系統的使用者，而不是它的工程師

### 1.2 代理人系統的三個核心本質

無論框架多複雜，代理人系統的本質可濃縮成三個觀念：

| # | 核心觀念 | 說明 |
|---|---------|------|
| ① | **Context Engineering** | 餵給 LLM 看什麼、不給它看什麼。最基礎也最關鍵的工程槓桿，決定代理人的行為邊界 |
| ② | **Harness（確定性外殼）** | 用確定性的殼（shell）包住 LLM 的機率性核心（core）。定義哪些規則是不可談判的，保護系統免於被 hack |
| ③ | **Loop with Verification** | 設計可被驗證的迴圈，讓代理人自主迭代。清楚的 verification criterion 是自主優化的前提 |

> 如果你連自己系統的 verification loop 在哪裡都看不到，你就永遠只是這個系統的使用者，而不是它的工程師。

### 1.3 為什麼選擇這兩個套件？

刻意選擇兩個「保持簡單」的套件作為切入點。它們的共通精神是：**代理人系統的「程式碼」，其實是 Markdown。**

| 套件 | 作者 | 特色 | 我們學什麼 |
|------|------|------|-----------|
| `karpathy/autoresearch` | Karpathy | 只有 3 個核心檔案，約 100 行的 program.md | Agentic Loop 的最小骨架 |
| `garrytan/gstack` | Garry Tan (YC CEO) | 23 個 slash command，每個都是純 Markdown skill | Skill-Based Orchestration |

> 呼應 Spec-Driven Development：prompt 是 micro-spec，program.md / SKILL.md 就是 macro-spec。

---

## PART 1：karpathy/autoresearch — Agentic Loop 的最小可運作原型

### 1.4 autoresearch 的三個核心檔案

| 檔案 | 角色 | 說明 |
|------|------|------|
| `prepare.py` | Harness | 固定常數、資料下載、BPE tokenizer、評估工具。**不可改動** |
| `train.py` | Probabilistic Core | GPT model + Muon/AdamW optimizer + training loop。**Agent 改這個** |
| `program.md` | Spec / Skill | Agent 的指令書。**Human 改這個**——這是 Spec，是你設計研究組織的地方 |

**關鍵設定**：
- 每次訓練執行固定 5 分鐘 wall-clock
- 評估指標：val_bpb（越低越好，與 vocab size 無關），定義在 prepare.py

**Agent 的工作循環**：改 train.py → 跑 5 分鐘 → 看 val_bpb 是否下降 → 保留或捨棄 → 下一輪

### 1.5 為什麼這個設計是「神作」？

| autoresearch 元件 | AI Native 概念 | 意義 |
|-------------------|---------------|------|
| train.py | Probabilistic Core | LLM 可自由修改的區域，允許創造性發揮 |
| prepare.py | Deterministic Harness | 不給 Agent 碰——保證資料、評估一致 |
| val_bpb + 5-min budget | Verifiability Criterion | 清楚、可計算、跨實驗可比較的 loss function |
| program.md | Spec / Skill | 人類「程式設計」代理人組織的地方 |
| 通宵跑 100 次實驗 | Snowball / Incremental Loop | 每次迭代都有可執行交付物 |

> 這五個元素湊在一起，就是一個最小可運作的 Agentic Research Loop。簡單、可理解、可複現——這才是最強的設計。

### 1.6 「知其所以然」的四個細節

1. **為什麼固定 5 分鐘？** — 固定時間預算讓所有實驗可直接比較，用 wall-clock 當 normalizer 是控制變因的工程實踐
2. **為什麼用 val_bpb 而不是 val_loss？** — Agent 可能修改 vocab_size；bits per byte 讓不同 tokenizer 設定下的實驗公平比較（評估設計的深思熟慮）
3. **為什麼 prepare.py 不能被 Agent 修改？** — Harness 核心原則：若資料處理也可被 Agent 改，它可能用「讓 eval set 變小」來作弊降低 val_bpb。Harness 定義「不可談判的規則」
4. **為什麼 program.md 故意保持簡陋？** — 這不是答案，而是可迭代的起點。調整 program.md 的過程就是在設計你自己的研究組織

> **val_bpb 設計的經典性**：若 Karpathy 用 per-token loss 當評估指標 → Agent 會發現「把 vocab_size 改大就能降 loss」→ reward hacking → Overfitting。改用 val_bpb（byte-level normalization）→ Agent 就算改 vocab_size，也得真的讓模型變好才能贏。這是 Verifiability Mindset + Harness Engineering 的經典。

### 1.7 給研究生的五步移植法

| autoresearch 元件 | 你的研究任務對應項 |
|-------------------|------------------|
| train.py | 你的待優化模組（研究的「被動部分」） |
| val_bpb | 你研究任務的驗證準則 |
| program.md | 你寫給 Agent 的研究 SOP |
| 5-min budget | 你定義的終止條件 |
| prepare.py 防改 | 你的 Harness 邊界 |

> **核心問題**：「我手上的研究任務，有哪一塊可以讓 Agent 通宵幫我跑？」這個問題的答案，決定了你跟同齡研究生的產出差距。

---

## PART 2：garrytan/gstack — 用 Markdown 組一支「虛擬工程團隊」

### 2.1 gstack 概述

gstack 是 Y Combinator 總裁 Garry Tan 開源的 Claude Code 設定檔。核心 claim：一個人加上這套工具，可以像 20 人的團隊一樣出貨。整個系統本質上就是一堆 Markdown 檔案組成的 slash commands，每個對應一個「虛擬角色」。

| Slash Command | 角色 |
|---------------|------|
| `/office-hours` | 扮演 YC Office Hours，用 6 個「逼問」問題重構產品想法 |
| `/plan-eng-review` | 扮演工程主管，鎖定架構、資料流、diagram 與邊界情況 |
| `/cso` | 扮演資安長，跑 OWASP Top 10 + STRIDE 威脅模型 |
| `/ship` | 扮演 Release 工程師，同步 main、跑測試、push、開 PR |

### 2.2 gstack 揭示的三個核心

1. **代理人是「工作流程」而不是「全能 AI」**
   - 整體流程：Think → Plan → Build → Review → Test → Ship → Reflect
   - 每個 skill 嚴格負責一個階段，下一個 skill 讀前一個 skill 的輸出——教科書等級的 Pipeline + Context Passing

2. **Skills 比 Sub-Agents 更好用（通常）**
   - 23 個 skill 都是純 Markdown，沒有一個是會自主迴圈的 agent
   - 組合起來效果卻與多 agent 系統相當
   - 真正的智慧來自流程編排（workflow orchestrator），而不是 agent 自主性

3. **標準化代理人可讀文件**
   - AGENTS.md、CLAUDE.md、SKILL.md 是 2025–2026 年浮現的代理人可讀文件標準
   - 你的 repo 除了給人看的 README，還要有給 Agent 看的 CLAUDE.md

> **Context Engineering 的核心**：代理人看得到什麼、看不到什麼，才是系統的主要工程槓桿。gstack 用「檔案作為 context 的載體」實現了這件事——簡單、可除錯、可版本控管。

### 2.3 autoresearch vs gstack：兩個維度的對照

| 維度 | autoresearch | gstack |
|------|-------------|--------|
| 時間軸 | 縱向迭代（同一任務跑 100 次） | 橫向階段（think → plan → ship） |
| 核心抽象 | Loop + Verifier | Pipeline + Context |
| Agent 角色 | 一個 agent 通宵優化 | 23 個 skill 依序協作 |
| Harness 類型 | 固定 prepare.py + 固定 5-min | 固定 skill order + 固定 context 傳遞 |
| 適合學習 | Verifiability Mindset, Self-improving loop | Skill Design, Workflow Orchestration |

- **autoresearch** 教你怎麼讓 Agent 在**縱深**上迭代——反覆優化同一個任務直到收斂
- **gstack** 教你怎麼讓 Agent 在**橫向**上協作——多個 skill 串連完成完整工作流程

---

## PART 3：為什麼 CSer 應該從這兩個套件入手？

### 3.1 建立「系統視角」而不是「套件使用者」視角

- 常見陷阱：看到 paper 附的 repo 就直接 pip install 用，把 LangChain 當標準答案
- 讀 autoresearch 的 100 行 program.md，比讀 LangGraph 的 10,000 行原始碼，更能理解「Agent 是怎麼運作的」

### 3.2 寫論文時必須解釋清楚「為什麼這樣設計」

- Reviewer 會問：你的 verification criterion 是什麼？你如何防止 reward hacking？你的 Harness 邊界在哪？
- 只用黑盒子框架，這些問題都答不出來

### 3.3 研究工作本身越來越像 autoresearch

- Karpathy 的未來預言：研究是 AI agent 群的事情
- 現在就可以問自己：「我手上的研究任務，有哪一塊可以讓 Agent 通宵幫我跑？」

### 3.4 核心概念對應表

| 課程概念 | autoresearch 對應 | gstack 對應 |
|---------|------------------|------------|
| Spec-Driven Development | program.md 就是 spec | — |
| Eval-Driven Development | val_bpb 就是評估 | — |
| Harness Engineering | prepare.py 防改 + 固定 5-min | skill 順序鎖定 |
| Verifiability Mindset | 5-min budget + 固定 eval | — |
| Snowball Development | 通宵 100 次迭代 | — |
| Context Engineering | — | skill 間的 context 傳遞 |

### 3.5 結語

- 如果你只想當 Agentic 系統的**使用者** → 直接裝 OpenClaw 就好（直到出錯的那一天）
- 如果你想當 Agentic 系統的**工程師**（甚至研究者） → 請從 autoresearch 和 gstack 開始——小到你可以完整讀完，然後真正理解

> 代理人時代最稀缺的能力，不是「會用 AI 的人」，而是「能設計 AI 協作系統的人」。知其然而知其所以然——這句話在 Agentic 時代變得更重要了。

### 3.6 設計你自己的 Agentic Flow——「五步移植法」

| Step | 內容 | 說明 |
|------|------|------|
| Step 1 | **找到你的問題** | 有沒有「反覆做、每次都想做得更好、但沒有時間」的任務？（論文回顧、程式碼重構、競爭分析、教材設計） |
| Step 2 | **設計你的 val_bpb** | 品質指標必須可量化、與目標高度相關、可自動計算（覆蓋率、測試覆蓋率、完成率等） |
| Step 3 | **設計迴圈結構** | SETUP → ACTION → EVALUATE → DECIDE → TERMINATE |
| Step 4 | **設計 program.md** | Agent 指令文件須包含：目標（量化指標）、每輪動作、終止條件、輸出格式 |
| Step 5 | **定義 Harness 邊界** | 固定層（評分框架、資料來源 API、輸出格式模板）vs. 行動空間（搜尋策略、寫作內容、迭代優先級） |

### 3.7 延伸閱讀

- 📦 `karpathy/autoresearch` — Agentic Research Loop 最小原型
- 📦 `garrytan/gstack` — Skill-Based 虛擬工程團隊
- Karpathy 的「Dummy's Guide」（autoresearch 補充讀物）

---

## 第二部分：AI 正在抹掉職缺嗎？數據背後的職涯轉型真相

> 從 Stanford SDEL「煤礦金絲雀」研究看初階白領的結構性崩塌

### 4.1 三個衝擊性數據

| 數據 | 來源 | 說明 |
|------|------|------|
| **13–16%** 年輕勞工就業降幅 | Stanford SDEL (2025) | 22–25 歲在 AI 高暴露職業中的相對就業下降幅度 |
| **~50%** 新鮮人聘用下降 | SignalFire (2025) | 前 15 大科技公司對工作經驗 <1 年者的聘用數 2019–2024 年降幅 |
| **27.5%** 程式設計師職缺蒸發 | U.S. BLS / IEEE Spectrum | 美國「Programmers」整體就業 2023–2025 年降幅；同期「Software Developers」僅降 0.3% |

> AI 不是在「裁員」——而是在系統性地「不請人」。

### 4.2 研究來源：「煤礦裡的金絲雀」

- **論文**：Canaries in the Coal Mine? Six Facts about the Recent Employment Effects of AI
- **作者**：Erik Brynjolfsson、Bharat Chandar、Ruyu Chen（Stanford Digital Economy Lab）
- **數據基礎**：ADP 工資單數據，涵蓋數百萬勞工，月度更新至 2025 年 9 月
- 2026 年補充論文進一步澄清時序：顯著效應從 2024 年開始
- **方法論**：使用 firm-time fixed effects 排除經濟週期與利率影響；大樣本 Panel Data + Eloundou/Tomlinson AI 暴露度量

### 4.3 六大核心發現

1. **年輕暴露者就業下降** — 22–25 歲在高 AI 暴露職業的相對就業下降 13–16%
2. **年長勞工不受衝擊** — 同一職業中，經驗豐富者就業持平甚至增長
3. **調整透過就業數、非薪資** — 企業並沒有降薪，而是不回補空缺——靜悄悄地移除入口
4. **自動化 > 增強** — 下降集中在 AI 用於「取代」而非「輔助」的任務類型
5. **結果對多種替代解釋穩健** — 排除科技業、遠距工作等因素後仍顯著
6. **結構性而非週期性** — 非利率、非景氣循環造成——這是長期趨勢的起點

> AI 不是在「裁員」，而是在「不請人」。企業悄悄地、系統性地移除了職涯階梯的最底層。

### 4.4 多個獨立數據源指向同一結論

| 來源 | 發現 |
|------|------|
| SignalFire 2025 | 大科技公司新鮮人聘用下降 50%（2019–2024） |
| EY 報告 2025 | 印度 IT 服務業初階職位減少 20–25% |
| LinkedIn / Indeed / Eures | 歐洲主要國家 2024 年初階科技職位下降 35% |
| WEF 未來就業 2025 | 2025–2030 年間，全球 22% 的現有職位將消失或大幅改造；39% 的核心技能將改變 |

### 4.5 為什麼是年輕人？隱性知識的經濟學

| 顯性知識 (Explicit Knowledge) | 隱性知識 (Tacit Knowledge) |
|------------------------------|--------------------------|
| 寫 CRUD API、撰寫標準化 marketing copy、基礎財報分析 | 如何在會議中讀出客戶真實需求、如何在 code review 時看出架構問題、如何跟難搞的同事協作 |
| 教科書、規章、SOP、可標準化輸出 | 需要「在場」、「試錯」、「師徒傳承」 |
| **這正是 LLM 在海量文本上訓練而來的最強項，也正好是剛畢業學生最引以為傲的能力** | **無法被編碼，因此難以被 AI 取代** |

> Polanyi (1966)：We can know more than we can tell. David Autor (2014) 將此命名為 Polanyi's Paradox。年輕人的比較優勢正好是顯性知識的新鮮庫存，而這正是 AI 最擅長替代的。

### 4.6 另一面：AI 對已雇用年輕人的加速效果

- Brynjolfsson, Li & Raymond (2023/2025, QJE)：Generative AI at Work — 5,172 位客服人員真實實驗
  - AI 輔助帶來 14–15% 平均生產力提升
  - 新手與低技能者提升達 **34%**，資深員工幾乎無提升
- **真正的解讀**：對已被僱用的年輕人，AI 是最好的加速器；但僱用的門檻因為 AI 而被大幅提高
- 「入職越來越難，但一旦進去成長越來越快」——這是一個高度兩極化的賽局

### 4.7 職業階梯的斷裂：「吃掉自己的種子」

- **傳統職涯模型**：新鮮人做搬磚 → 累積隱性知識 → 晉升中階管理者（線性晉升）
- **AI 時代新現實**：AI 直接處理「搬磚工作」（簡單 coding、基礎分析、郵件草稿）→ 企業短期獲利 → 但長期災難：未來 10 年後，誰來當中階管理者？誰懂底層系統？

> Companies stopping junior hiring in 2025 are effectively eating their own seed corn. — Rezi.ai

- IBM 的反向押注：2025 年宣布將年輕聘用人數擴大 3 倍，正是擔心中階人才池枯竭

### 4.8 IEEE Spectrum 微觀數據：不是所有職位都一樣慘

| 職位類型 | 就業變化 (%) |
|---------|------------|
| AI Engineers | **+18** |
| Information Security Analysts | **+15** |
| Software Developers | **-0.3** |
| Programmers（執行層） | **-27.5** |

> 在「AI 生成程式碼」的時代，純執行型 coding 正在被貶值。真正值錢的是：系統設計（Architecture）、需求拆解（Problem Decomposition）、安全性判斷（Security Judgement）。

### 4.9 「向下平權」的陷阱：生產力提升的分配不均

- **表面**：AI 壓縮了技能差距（新手提升 34%、資深提升 0%）
- **實際**：標準化產出的市場價值同步貶值——供給側一夕爆增。超額報酬集中在：算力持有者、模型持有者、戰略制定者
- **80 分的通貨膨脹**：如果你的工作可以被 AI 生成 80 分，你的薪資就會趨近於「80 分的通貨膨脹後市場價」

> 別把「我會用 AI」當成護城河。每個人都會用。真正的護城河是 AI 無法輕易複製的判斷力與信任關係。

### 4.10 Automation vs. Augmentation

- **Anthropic Economic Index 觀察**：
  - Claude.ai 用戶端：Augmentation 52% > Automation 45%（2025 年底）
  - API 流量：Automation 主導，且比重持續上升 → 企業級 AI 使用正往純自動化移動
- **你的個人選擇題**：
  - ❌ Automation 路線：把工作丟給 AI → 工作邊界縮小 → 第一個被裁的就是你
  - ✅ Augmentation 路線：用 AI 擴張能力邊界 → 一個人活成一支軍隊

### 4.11 職業 AI 暴露圖像

| 職業類別 | 理論暴露度 (%) |
|---------|-------------|
| 電腦與數學 | 94.3 |
| 商業與財務 | 94.3 |
| 管理 | 91.3 |
| 辦公室行政 | 90 |
| 法律 | 89 |
| 建築與工程 | 84.8 |
| 餐飲 | 16.9 |
| 農業 | 15.7 |
| 交通運輸 | 12.1 |
| 場地維護 | 3.9 |

> 資工系學生請注意：你們所在的「電腦與數學」類別，理論暴露度高達 94.3%。這代表你必須做出差異化。

### 4.12 薪資動態：「靜默調整」的真相

- Anthropic (2026)：未觀察到大規模失業
- Stanford SDEL：調整主要透過「就業（不補缺）」而非「薪資」——靜悄悄地發生
- **J-Curve 的警示**（Brynjolfsson, Rock & Syverson, 2021）：新技術的紅利有滯後，但初期調整成本落在最弱勢族群身上

> 此刻正是你們畢業的 2026–2028——恰好是調整成本最集中的時間窗口。這不是悲觀，是你必須提前備戰的理由。

### 4.13 四個職涯行動建議

#### 動作一：把 AI 當「陪練」而非「代筆」

- ❌ 錯誤：把作業丟給 ChatGPT → 貼上繳交
- ✅ 正確：
  - Spec-Driven Development：先寫清楚規格 → 讓 AI 產出 → 你做驗證與 code review
  - 讓 AI 當苛刻的客戶：「假裝你是 PM，嚴格 review 我的 HW3 RAG 設計」
  - 讓 AI 當敵對評審：「把這份架構提案當論文 reviewer 來撕」
- 目標：訓練品質控制權與邏輯敏銳度——隱性知識的核心

#### 動作二：從「線性晉升」轉向「職業網格（Career Grid）」

- 線性晉升：CS → 後端工程師 → 資深 → 技術主管
- 職業網格：基礎 × 領域 × 寫作 × 研究 ＝ 倍增價值
- AI 賦能網格策略：極速學習新領域、降低跨界門檻
- 對應：MCP、Agent 工具鏈、Skills 系統就是「你個人能力的跨域連接器」

#### 動作三：建立「線下非標準化」槓桿

| 賽道 | 說明 |
|------|------|
| **物理在場** | 田野研究、實驗室操作、客戶現場 debug。身體化知識無法遠端傳輸 |
| **戰略樞紐** | 跨部門協調、產品方向定義、組織變革。AI 給建議，但你要負責 |
| **社交信任** | 長期客戶關係、團隊文化建立、師徒傳承。沒有人能下載別人對你的信任 |

> WEF 2030 最關鍵的「成長中」軟實力：創造與批判思考、韌性與敏捷、好奇心與終身學習、領導與社會影響力。

#### 動作四：掌握 Harness Engineering 的制高點

| Harness 層 | 職場類比 |
|-----------|---------|
| 🔵 Input Harness | 像你的簡歷／面試答辯——能否清楚規格化自己的能力？ |
| ⚙️ Process Harness | 像你的工作流程——能否寫出自己的 SOP / Skill.md？ |
| ✅ Output Harness | 像你的交付物品質——能否建立自動化驗證機制？ |

> 核心原則：Deterministic shell wrapping a probabilistic core. 職涯版翻譯：你的人類判斷（確定性外殼）× AI 的生成能力（機率性核心）= 你的職場護城河。

### 4.14 誰擁有你的身份？

如果你的所有 coding 能力、分析能力、寫作能力都是 AI 給的，那你剩下什麼？你的識別性必須來自 AI 無法複製的部分：

- **好奇心（Wonder）**：問 AI 不會自己生成的問題——因為你活過、你困惑過、你在乎過
- **意志（Agency）**：在不確定中做出承諾與選擇——AI 給選項，但只有你能負責
- **品味（Taste）**：在海量生成內容中判斷什麼是好的——這需要你有真實的審美與價值觀

### 4.15 三句話總結

1. **別在 AI 擅長的賽道跟它競爭** — 去做它做不到的事：判斷、連結、創造、在場。你的比較優勢不在執行速度，在人類的不可化約性。
2. **「代筆」與「陪練」是兩種截然不同的人生軌跡** — 選擇哪一種，是你每次打開 ChatGPT 的那一刻就在決定的事。習慣的複利會決定你三年後站在哪裡。
3. **AI Native 時代的工程師思維** — Spec-Driven、Eval-Driven、Harness Engineering——這些是未來 10 年面對任何新技術都適用的 Meta 能力。

### 4.16 參考文獻

**一、核心實證研究**
1. Brynjolfsson, E., Chandar, B., & Chen, R. (2025). Canaries in the Coal Mine? Stanford SDEL.
2. Brynjolfsson, E., Chandar, B., & Chen, R. (2026). Canaries, Interest Rates, and Timing. Stanford SDEL.
3. Brynjolfsson, E., Li, D., & Raymond, L. R. (2025). Generative AI at Work. QJE, 140(2), 889–942.

**二、Anthropic Economic Index 系列**
1. Handa, K. et al. (2025). Which Economic Tasks are Performed with AI?
2. Anthropic (2025). Economic Index Report V3.
3. Anthropic (2026, Jan). Economic Index Report: Economic Primitives.
4. Anthropic (2026, Mar). Economic Index Report: Learning Curves.

**三、理論基礎與產業報告**
1. Autor, D. H. (2014). Polanyi's Paradox. NBER No. 20485.
2. Polanyi, M. (1966). The Tacit Dimension.
3. WEF (2025). Future of Jobs Report 2025.
4. Brynjolfsson, E., Rock, D., & Syverson, C. (2021). The Productivity J-Curve. AEJ: Macroeconomics.
5. ICLE (2026). AI, Productivity, and Labor Markets: A Review.

---

## 第三部分：The Agentic Stack — 在名詞的海裡找座標

> 從 OS 本質重新理解 Skill / MCP / A2A / CLI Agent

### 5.1 Vibe Coding 已退役，現在叫 Agentic Engineering

「Vibe Coding」從 Karpathy 在 2026 年 2 月 4 日的一則貼文開始。AI 輔助開發已出現本質性轉變：

| | Vibe Coding | Agentic Engineering |
|---|-----------|-------------------|
| 對象 | 不會寫程式的人 | 專業工程師 |
| 任務 | 替所有人拉高地板 | 替專業人士守住天花板 |
| 行為 | 全程憑感覺 | 寫 spec、跑測試、code review、管版本、監控生產 |
| 風險容忍 | 週末小專案 | 對 P&L 負責的系統 |

### 5.2 破題：混亂的名詞海

- 這個領域的術語現況極度混亂：Skills / Extensions / Tools、MCP servers / Function calling、Sub-agents / A2A protocol、Agent Cards / Gateways / Hooks…
- 每個詞都有「官方定義」，但來自不同公司、不同文件、不同時期
- **沒有人告訴你它們之間是什麼關係**——這不是你的問題，這是這個領域的現狀

### 5.3 真相與立場

**真相**：大家都在爭奪 Agentic OS 的地位
- 各自為政的標準：Anthropic 推 MCP（2024 Q4）、Google 推 A2A（2025 Q1）
- 連廠商自己都在演進：連 Anthropic 自己的 Claude Code 和 Claude API 對「skill」的定義都在持續演進
- 盲人摸象的困境：大家都在描述大象的不同部位，沒有人告訴你這是一隻大象

**立場**：
- ✅ 追求：看到任何新名詞能立刻知道它活在哪一層、解決什麼問題；能親手設計每一層的選擇；理解設計決策之間的 trade-off
- ❌ 不追求：記住最新版本的 API 細節；盲目套用「業界最佳實踐」；知道哪個框架「最強」

> 工程師的真正能力：不是「會用哪個工具」，而是「能在 5 分鐘內拆解任何工具」的框架能力。這能力來自你大二學過的作業系統課程。

### 5.4 回到本質：這是 OS 的 IPC 問題

Agent 系統在做什麼？
- 有多個「執行單元」（agent）
- 它們需要交換資訊、協調執行
- 需要管理資源（context window、token budget、權限）

> 這不就是作業系統嗎？Process 管理、IPC、資源調度、權限隔離——OS 40 年前就在解決這些問題。

### 5.5 Intra-Process vs Inter-Process

| | Intra-Process（單一 process 內） | Inter-Process（跨 process） |
|---|-------------------------------|---------------------------|
| **OS** | Function call · Shared memory · Thread sync | Pipe/FIFO · File+lock · Socket · RPC |
| **Agent 對應** | Tool call（同一個 LLM loop 內）、Skill（同一個 agent 的能力擴充）、Extension hook | Sub-agent（parent-child pipe）、Mailbox（file IPC）、Gateway（WebSocket hub）、A2A Protocol（跨信任邊界的 RPC） |

> 這不是新學科，是你大二作業系統的延伸應用。

### 5.6 本質問題永遠是這幾個

1. **誰在控制執行流程？** — 確定性程式碼 vs 概率性 LLM
2. **資源怎麼共享／隔離？** — Context window、token budget 是共用的還是隔離的？
3. **同步還是非同步？** — Tool call 要 block 嗎？Agent 之間要等回應才繼續嗎？
4. **信任邊界在哪裡？** — 對方是你自己開的 sub-agent，還是陌生的第三方？
5. **失敗時怎麼恢復？** — Session replay？Checkpoint？還是重頭來過？

---

## 5.7 四大架構分類

用「哪種 IPC」來分類，混亂的名詞立刻清晰。**這四類不是競爭關係，是不同層的東西——一個 agent 可以同時用四種機制。**

| 類別 | OS 對應 | 代表方案 | 本質 |
|------|--------|---------|------|
| 1. **Skill** | Library / Plugin | Claude Skills、Pi Skills | Agent 內部能力擴充 |
| 2. **MCP** | System Call / RPC | Anthropic MCP | Agent → 外部 Tool（垂直整合） |
| 3. **A2A** | Network Service / IPC | Google A2A Protocol | Agent ↔ Agent 水平協作 |
| 4. **CLI Agent** | Shell Process | Pi、Claude Code CLI | Runtime 本身 + stdio |

### 類別 1：Skill — Agent 的內部能力

- **定義**：Agent 內部的能力擴充——住在你 repo 裡、和你 agent 一起生活
- **OS 類比**：函式庫（library），如 libc、numpy
- **控制流**：Deterministic——步驟寫死在 markdown / code 裡
- **執行位置**：Agent 本身的 process / loop
- **擴充方式**：放進 skills 目錄，runtime 自動發現並載入
- **典型例子**：「如何做 code review」、「如何寫 commit message」

> 這就是課程的 skill.md 傳統——一個 skill 是一份可執行的 spec。

### 類別 2：MCP — Agent 呼叫外部工具

- **定義**：Agent 透過標準協議呼叫外部 service（資料庫、SaaS、API）
- **OS 類比**：跨 process RPC（gRPC、dbus）
- **控制流**：LLM 決定何時呼叫，server 提供確定性回應
- **執行位置**：MCP server 是另一個獨立 process（可遠端）
- **協議**：JSON-RPC 2.0（over stdio 或 HTTP/SSE）
- **典型例子**：Gmail MCP、Google Drive MCP、自家資料庫 MCP

> MCP 是「垂直整合」—— agent 向下連接 tool。MCP 的設計精髓 = 把「能力」從 agent runtime 解耦出來，跟 OS 把 driver 從 kernel 解耦是同一招。

### 類別 3：A2A — Agent 之間水平協作

- **定義**：獨立存在的 agent 之間、跨信任邊界的協作協議
- **OS 類比**：Network service + 認證（HTTPS + OAuth）
- **控制流**：雙方各自自主——沒有「parent」主控關係
- **執行位置**：跨機器、跨網路、跨組織
- **協議**：JSON-RPC 2.0 + SSE + OAuth 2.0
- **Discovery**：`/.well-known/agent-card.json`

**MCP vs A2A 常被搞混**：
| | MCP | A2A |
|---|-----|-----|
| 方向 | Agent → Tool（垂直） | Agent ↔ Agent（水平） |
| OS 對應 | System Call | Network Service |
| 信任假設 | Tool 是你授權的 | 對方可能是「陌生人」 |

### 類別 4：CLI Agent — Runtime 本身作為介面

- **定義**：把 agent 整個包成一個 shell 指令，用 stdio / file 和外界溝通
- **OS 類比**：Shell process（grep、awk、jq——小而專注、可組合）
- **控制流**：外部 orchestrator（human 或 script）決定
- **協議**：JSONL over stdio、或純 stdin/stdout 文字
- **典型例子**：pi, claude（CLI mode）, aider

> Unix 哲學的延伸：`echo "task" | pi | jq .output | some-other-agent`

### 5.8 四大架構的 Stack 位置

| 類別 | Stack 位置 | 性質 |
|------|-----------|------|
| Skill / Extension | Intra-process（library、plugin） | Agent 內部 |
| MCP | Inter-process vertical（system call、RPC to service） | Agent → 外部工具 |
| A2A | Inter-process horizontal（network service、agent peer） | Agent ↔ Agent |
| CLI Agent | 這一切的容器（shell process、runtime） | 承載所有機制 |

### 5.9 決策樹：我該用哪一類？

| 決策問題 | 答案 |
|---------|------|
| 把可重複的工作流程包成能力？ | → **類別 1: Skill**（agent 內部、deterministic） |
| 操作外部系統（DB、SaaS、API）？ | → **類別 2: MCP**（另起 process、標準 tool 介面） |
| 跟「獨立存在」的 agent 協商（跨組織）？ | → **類別 3: A2A**（跨網路、需 Agent Card、需認證） |
| 把 agent 變成可 shell 組合的工具？ | → **類別 4: CLI Agent**（stdio 介面、Unix 哲學） |

**反模式提醒**：
- ❌ 用 MCP 做 A2A 該做的事（把其他 agent 當 tool 呼叫，忽略信任邊界）
- ❌ 用 Skill 做 MCP 該做的事（把外部 API 硬塞進 prompt 裡）
- ❌ 用 Sub-agent 做 Skill 該做的事（明明 deterministic 卻開 LLM loop）

### 5.10 真實組合例子：Code Review Agent Team

| 類別 | 在此例中的角色 |
|------|-------------|
| 類別 4: CLI Agent | 三個 Pi instances（Scanner · Analyzer · Reporter），各自獨立 CLI process |
| 類別 1: Skill | 每個 agent 裝自己的專業 skill（run-lint.md、severity-ranking.md、markdown-format.md） |
| 類別 2: MCP | 連接外部系統（GitHub MCP 讀取 PR diff + Slack MCP 送通知） |
| 類別 3: A2A | 對外提供服務：別家公司的 review bot 可透過 A2A 呼叫整個 team |

> 沒有「用哪一個」的問題，只有「每一層選哪一個」的問題。

---

## 第四部分：Pi Framework — 親手解構的教具

### 6.1 為什麼選 Pi？

| 工具 | 特性 |
|------|------|
| Claude Code | 完備、功能強、預設幫你做了所有選擇 |
| Cursor | IDE 整合、綁在編輯器裡 |
| AutoGen / LangGraph | Library 型、抽象層級高 |
| **Pi** | **極簡 CLI、刻意不做選擇** |

Pi 刻意的「不做」：
- ❌ No sub-agents —「有很多種做法，自己 spawn」
- ❌ No MCP —「用 extension 自己加」
- ❌ No permission popup —「跑在 container 裡，或自己寫」
- ✅ 只有 4 個原始 tool：read / write / edit / bash

### 6.2 四個原始 Tool 深度解析

#### Tool 1 · read — 讀檔

```json
{ "name": "read", "parameters": { "path": "string", "offset": "number", "limit": "number" } }
```
- 強制 offset / limit 是 Context Engineering 的具體展現
- 模型不能無腦讀整個 5MB log file，每次讀取都必須明確指定範圍
- 培養「只讀需要的部分」的工程直覺

#### Tool 2 · write — 整檔寫入

```json
{ "name": "write", "parameters": { "path": "string", "content": "string" } }
```
- **全檔覆蓋**，不是 append。要 append 必須 read → 拼接 → write
- 強制顯式的 read-modify-write 循環，避免「靜默修改」造成 race condition

#### Tool 3 · edit — 精準替換

```json
{ "name": "edit", "parameters": { "path": "string", "old_string": "string", "new_string": "string", "replace_all": "boolean" } }
```
- Token 效率：改一行不需要重寫整檔（10KB 檔案改一行 → 只送 20 字元 vs. 送 10KB）
- old_string 必須完全匹配，模型「亂改」會直接失敗——用 schema 強制精確性

#### Tool 4 · bash — 萬能逃生口

```json
{ "name": "bash", "parameters": { "command": "string", "timeout": "number" } }
```
- 任何 Pi 沒有內建的能力，都可以透過 bash 達成
- 用一個工具取代所有專用工具：`glob_search` → `find`、`grep_tool` → `grep`、`run_test` → `pytest`、`git_commit` → `git commit`…

**「萬能 bash」的取捨**：

| 好處 | 代價 |
|------|------|
| 極簡 schema，system prompt 小 | 模型必須自己組指令，弱模型容易出錯 |
| 任何能力都能透過組合 bash 達成 | 沒有結構化 schema → output 解析較難 |
| 工具數量少 → 模型選擇負擔低 | 安全風險高 |

> 這就是「設計選擇的可見化」：你看見了取捨，而不是隱藏在「Magic」背後。

### 6.3 教學哲學：從「使用」到「親手建造」

| 你現在的狀態（使用者） | 我們要去的狀態（工程師） |
|----------------------|----------------------|
| 打開 Claude Code，專心寫程式 | 打開 Pi，開始理解極簡核心 |
| MCP? → 預設支援 | 自己實作 extension |
| Sub-agent? → 內建 | 用 tmux + bash 自行 spawn |
| A2A? → 不支援 | 設計 mailbox 通訊 |

> Pi 不是最好的架構。它是最好的教具。

### 6.4 Headless 的兩層語義

| 第一層：基建意義 | 第二層：架構意義 |
|---------------|---------------|
| 核心精神：沒人盯著它 | 核心精神：核心可被任意組裝 |
| Headless server、Headless browser、CLI --non-interactive | Headless CMS、Headless Commerce、Headless UI |
| 設計動機：自動化、CI、腳本化 | 設計動機：API-first、composable、不綁定特定前端 |

**Headlessness 光譜**：

| 工具 / 模式 | Head | Body | Headless 程度 |
|------------|------|------|-------------|
| Framework-first 工具 | 全包，UX 鎖死 | 隱藏在框架內 | ❌ 不 headless |
| Claude Code 預設模式 | 附 TUI、slash commands | engine 由 Anthropic 管 | 🟡 半 headless |
| Claude Code `-p` print mode | head 拔掉 | engine 同上 | ✅ Headless agent |
| Pi Framework | 沒有 head | 連 loop 都你自己 wire | ✅✅ Headless core |

> Headless ＝ body 與 head decoupling 的設計哲學。先 headless，再 head。唯有先看清 probabilistic core 的真實樣貌，才能判斷任何 head 是在增值還是在遮蔽問題。

### 6.5 Pi 全貌：極簡 Agent Kernel + 五種擴充機制

#### 四種執行模式

| 模式 | 指令 | OS 類比 | 使用時機 |
|------|------|--------|---------|
| Interactive | `pi` | 互動式 shell (bash) | 人類坐在終端前操作 |
| Print | `pi -p "..."` | 一次性指令 (ls, grep) | Script 裡呼叫、CI/CD |
| JSON | `pi --mode json` | 結構化輸出 (ls --json) | 下游程式解析事件流 |
| RPC | `pi --mode rpc` | Daemon process + IPC | 被另一個 process 長期驅動 |

SDK 補充：`@mariozechner/pi-coding-agent` npm 套件 → 可在 Node.js 裡 import 使用

#### 五種擴充機制（從最簡到最強）

| 機制 | 說明 | OS 類比 |
|------|------|--------|
| **Prompt Template** | Markdown 檔，可重用的 prompt（帶參數） | Shell alias |
| **Skill** | SKILL.md + 支援檔，「如何做 X」的 spec | Library / plugin |
| **Extension** | TypeScript module，改 runtime 行為（tool / hook / UI） | Kernel module |
| **Pi Package** | package.json 宣告，把以上打包分發 | APT package |

#### Extension 能力（Pi 的「kernel module」）

- 自訂 tool：加新 tool，或替換掉內建的 4 個 tool
- Sub-agent 實作：Pi 本身沒有 sub-agent，你用 extension 自己做
- Permission gate：加權限檢查、路徑保護
- 客製 UI：狀態列、頁首頁尾、浮動視窗
- MCP 整合：加上 MCP 支援（Pi 本身不內建）
- Git checkpointing：每次 tool call 前自動 git commit

```typescript
export default function (pi: ExtensionAPI) {
  pi.registerTool({ name: "deploy", ... });
  pi.registerCommand("stats", { ... });
  pi.on("tool_call", async (event, ctx) => { /* intercept & modify */ });
}
```

### 6.6 Session Tree：Pi 的版本控制

Pi 的 session 是一棵樹，不是一條線。儲存格式是 JSONL 檔，每筆訊息有 id 和 parentId。

| 指令 | 功能 | CS 對應概念 |
|------|------|-----------|
| `/tree` | 進入 session tree 瀏覽器 | Git commit graph |
| `/fork` | 從目前分支建立新 session 檔 | Git branch |
| `/compact` | 把舊訊息壓縮摘要 | Git shallow clone / snapshot |

> Session tree 是「agent context 的版本控制」——所有 checkpoint、錯誤嘗試、平行探索都保留得住。這是 Verifiability Mindset 的基礎建設。

### 6.7 Context Files 與 Skill 標準

**啟動時自動載入順序**：`~/.pi/agent/AGENTS.md` → `../../AGENTS.md` → `../AGENTS.md` → `./AGENTS.md`（從 cwd 往上找）

- Pi 遵循 Agent Skills 標準（agentskills.io）
- Skill = 一個資料夾，內含 SKILL.md + 支援檔
- 也相容 CLAUDE.md——這不是偶然，是業界正在形成的 convention

### 6.8 Message Queue：Pi 的併發模型

| 操作 | OS 類比 | 用途 |
|------|--------|------|
| Enter — Steering 訊息 | SIGINT + 新指令 | 現在的 tool 跑完就插隊，用於即時方向修正 |
| Alt+Enter — Follow-up 訊息 | 排進 job queue 尾端 | 等 agent 整個完成才執行，用於非緊急的後續指令 |
| Escape — 中斷 + 還原 | SIGKILL + 狀態還原 | 中斷 + 把未送出訊息還給 editor，用於緊急停止 |

> 背後是 producer-consumer pattern + priority queue。Agent 是 consumer，人類是 producer，queue 有兩種優先級（steering > follow-up）。

### 6.9 Pi 完整功能 ↔ OS 概念終極對照表

| Pi 功能 | OS / CS 對應 | Pi 功能 | OS / CS 對應 |
|--------|-------------|---------|-------------|
| Interactive mode | 互動式 shell | Theme | Terminal PS1 / dotfiles |
| Print mode | 一次性指令 | Pi Package | APT package |
| JSON mode | 結構化輸出 | Session (JSONL) | Database transaction log |
| RPC mode | Daemon + IPC | /tree | Git commit graph |
| SDK | Library binding | /fork | Git branch |
| 4 built-in tools | 最小 syscall 集合 | /compact | Snapshot / checkpoint |
| Prompt Template | Shell alias / macro | AGENTS.md | .bashrc / .profile |
| Skill | Library / plugin | Message Queue | Producer-consumer + priority |
| Extension | Loadable kernel module | pi.on("tool_call") | Kernel event hook |

### 6.10 Pi 如何映射四大架構

| 架構類別 | Claude Code 的做法 | Pi 的做法 |
|---------|-------------------|----------|
| 類別 1: Skill | ~/.claude/skills/ 內建 | Skill 機制（遵循 agentskills.io 標準） |
| 類別 2: MCP | 原生支援 | Extension 裡橋接（或用 bash 直接呼叫 MCP server） |
| 類別 3: A2A | 無（封閉生態） | 多個 Pi instances + 自訂 transport |
| 類別 4: CLI Agent | claude CLI | Pi 本身（原生 --mode rpc 的 JSONL 協議） |
| Sub-agent | Task tool | Extension 自己實作（bash + tmux 或 SDK spawn） |
| Permission | 內建 popup | Extension 自己做 |

### 6.11 三個核心訊息

1. **現在的 agent 名詞海是暫時的** — 沒有大一統架構，但 OS 早就統一過一次了。你的羅盤是作業系統課本，不是某家公司的最新 blog post
2. **Skill / MCP / A2A / CLI 不是競爭關係** — 它們解決不同層的問題。對的問題是「每一層該選什麼」
3. **最好的架構 ≠ 最好的教具** — Pi 不如 Claude Code 好用，但它逼你親手做每個決定。能力來自理解，不來自工具

---

## 第五部分：Pi + Gemma 4 本地端 Coding Agent 實戰

> 把 Pi 真的跑在你的筆電上——從「概念」到「肌肉記憶」

### 7.1 本週目標

- 親手把 Gemma 4 跑在自己的機器上、用 Pi 連接它並設計 Skill 與 Extension
- 三個面向：核心哲學（Skill vs Sub-Agent）、架構層（IPC 五層模型）、實作與反思（弱模型暴露的 Harness 必要性）

**來源文獻**：Patrick Loeber, "How to run a local coding agent with Gemma 4 and Pi" (2026-04-27)

### 7.2 工作坊整體架構

| 元件 | 角色 | 對應課程概念 |
|------|------|-----------|
| LM Studio | 本地模型伺服器 | LLMOps Layer 2 |
| Gemma 4 26B A4B | 推理核心 | MoE 架構、Apache 2.0 |
| Pi Agent | Terminal harness | Skill 哲學的極致 |
| Skills | 動態能力包 | Markdown spec → 載入 |
| Extensions | 安全與擴充 | Verifiability Mindset |

### 7.3 Gemma 4 模型概覽

| Model | Architecture | Context | 適用場景 |
|-------|-------------|---------|---------|
| Gemma 4 E2B | Dense | 128K | Edge / Raspberry Pi |
| Gemma 4 E4B | Dense | 128K | 學生筆電起點 |
| **Gemma 4 26B A4B** | **MoE (4B active)** | **256K** | **本週主推 ★** |
| Gemma 4 31B | Dense | 256K | 高階工作站 |

> **MoE 核心觀念**：26B-A4B = 「26B 的品質 × 4B 的速度」——每次推理只 activate 4B 參數，但模型整體知識量達到 26B 等級。

**MoE 的記憶體陷阱**：計算量 ≈ 4B 模型等級（快），但 VRAM 佔用 ≈ 26B 模型等級（所有 expert 參數都必須常駐記憶體）

### 7.4 量化選擇與 Context Size

**量化格式**：

| Quantization | Size | Quality | 適合誰 |
|-------------|------|---------|--------|
| Q4_K_M | 18 GB | Good balance | 多數學生（本週預設） |
| Q6_K | 24 GB | Higher quality | RTX 3090 / 4090 |
| Q8_0 | 28 GB | Near-original | 高階工作站 |

**Context Size vs VRAM**：

| Use Case | Context | 額外 VRAM |
|----------|---------|----------|
| Small edits | 16K | ~1 GB |
| Standard coding | 64K | ~4 GB |
| Multi-file refactors | 128K | ~8 GB |
| Full repo | 256K | ~16 GB |

### 7.5 實作環節：兩條路徑

#### 路徑 A：本地端跑 Gemma 4（VRAM ≥ 8GB）

1. 安裝 LM Studio → 下載 gemma-4-26b-a4b (Q4_K_M)
2. 啟動本地 API Server，確認端點：`curl http://localhost:1234/v1/models`
3. 安裝 Pi：`npm install -g @mariozechner/pi-coding-agent`
4. 設定 `~/.pi/agent/models.json`：

```json
{
  "providers": {
    "lmstudio": {
      "baseUrl": "http://localhost:1234/v1",
      "api": "openai-completions",
      "apiKey": "lm-studio",
      "models": [{ "id": "google/gemma-4-26b-a4b", "input": ["text", "image"] }]
    }
  }
}
```

5. 執行 `pi` 並輸入 `/model lmstudio:google/gemma-4-26b-a4b`

#### 路徑 B：課程 LiteLLM Gateway（無 GPU 也可）

```json
{
  "providers": {
    "aiase-litellm": {
      "baseUrl": "https://litellm.aiase.ncku.edu.tw/v1",
      "api": "openai-completions",
      "apiKey": "<課程提供的 API key>",
      "models": [
        { "id": "claude-sonnet-4-7", "input": ["text", "image"] },
        { "id": "gemini-2-flash", "input": ["text", "image"] }
      ]
    }
  }
}
```

> 對照觀察：models.json（應用層，單一 agent 用）vs. LiteLLM config.yaml（基礎設施層，可共用，含預算管理與完整 logging）

### 7.6 共同任務

| 任務 | 內容 | 觀察重點 |
|------|------|---------|
| **任務 1：Hello World** | 請 Pi 建立 hello.py | Pi 用了哪個 tool？Schema 結構？System prompt 大小（/debug 查看）？ |
| **任務 2：安裝並使用 Skill** | `git clone pi-skills` → `/skill:liteparse` 解析 PDF | Skill 本質是注入到 system prompt 的 spec 片段；動態載入不汙染預設 context |
| **任務 3：Permission Gate Extension** | `pi --extension permission-gate.ts`，請 Agent 執行含 rm 的任務 | Pi 預設 YOLO（直接執行所有 bash command）；permission gate 攔截危險指令 |
| **任務 4：Session Tree 操作** | `/tree`、`/fork`、`/compact` 對照實驗 | 從同一起點 /fork 三個分支 → 比較結果 → 整合最佳解（像 Git merge） |

### 7.7 核心洞察：弱模型 = 顯微鏡

> 在 Sonnet 4.7 上看不到的失敗模式，在 Gemma 4 上會放大成顯而易見的教學素材

| Gemma 4 的失敗模式 | 對應的工程教訓 |
|-------------------|-------------|
| 跳過 spec 階段直接動手 | SDD 強制階段化的必要性 |
| Tool-call 解析錯誤 | Schema Normalization 的重要性 |
| 為 Claude 設計的 prompt 失效 | Prompt portability is a myth |
| /compact 後仍 OOM | Context Engineering 是 RAM 管理 |

- **強模型**像優秀的員工：你交辦不清楚，他也會幫你補齊 → 你永遠不知道自己的 spec 有多爛
- **弱模型**像新手實習生：你說什麼他做什麼，你沒說的都會出錯 → 每一個失敗都是工程問題的精確定位

> 學生在弱模型上學會的工程紀律，回到強模型上會 10x productivity。這就是 Harness Engineering 的真正價值：不是讓弱模型變強，而是讓工程師看清楚自己的 harness 設計是否夠扎實。

### 7.8 串接課程概念

#### 串接 LLMOps

| LLMOps Layer 2 概念 | 今天的對應實作 |
|--------------------|-------------|
| 推論引擎 | LM Studio（內含 llama.cpp） |
| 量化格式選擇 | Q4_K_M / Q6_K / Q8_0 的取捨 |
| OpenAI-compatible endpoint | localhost:1234/v1 |
| MoE 架構 | Gemma 4 26B-A4B 的 active parameter 機制 |
| Context vs. VRAM 取捨 | 16K / 64K / 128K / 256K 選擇表 |

#### 串接 SDD

- 一個 Skill = 一份 Markdown spec，定義模型在該能力範疇內應有的行為邊界
- LLM 讀懂 Skill = LLM 讀懂 spec（sub-spec 注入到 LLM 的執行上下文中）
- Skill 觸發 = spec 凍結後派工
- `/skill:liteparse` 不只是載入工具，而是「凍結並執行一份 sub-spec」—— SDD-first 的精髓

### 7.9 回家練習預告：三種 Harness 對照賽

| # | 模式 | Harness 哲學 | 預期體驗 |
|---|------|------------|---------|
| 1 | Claude Code + CLAUDE.md | Maximalist Harness | 流暢、快速，模型自動補足許多細節 |
| 2 | Pi + Gemma 4 | Minimalist Harness | 較多手動介入，每個失敗都清晰可見 |
| 3 | 純 API loop（自己寫） | No Harness | 你必須親自當 orchestrator，感受 harness 的價值 |

### 7.10 Pi 教給你的真正能力

| 能力 | 說明 |
|------|------|
| **看懂其他框架** | 看 Claude Code 的 Task tool → 「這是 L4 sub-agent，包裝成 L3 tool call」；看 LangGraph 的 node → 「這是 L2 skill」 |
| **設計自己的系統** | 拿到需求 → 先分類是 L2/L3/L4 問題 → 每一層獨立選擇方案 → 知道選擇之間的 trade-off |
| **面對未來新框架** | 2028 年會有新名詞…名字會變，OS 本質不會變。你能在 5 分鐘內知道它活在哪一層 |

### 7.11 一句話總結

> Pi 不是教你寫 agent——Pi 是教你看穿所有 agent 框架的本質。而本地端跑 Pi + Gemma 4，就是把這個本質烙進你的肌肉記憶。

---

## 結語引用

> 200 人的公司 → 只有 12–17 個「槍管」，其餘 180+ 人都是「子彈」。AI 不是消滅工作——它把這條線，畫得清清楚楚。過去的槍管，協調 20 個工程師。未來的槍管，協調 20 個 AI agent。「沒人告訴我怎麼做時，我自己決定怎麼做」——這個本質從沒變過。
>
> — Keith Rabois, Lenny's Podcast

---

## 附錄：延伸閱讀依架構層分類

| 層次 | 資源 |
|------|------|
| **Skill / Extension 層 (L2)** | Anthropic Claude Skills docs、Pi extensions API docs |
| **MCP 層 (L3)** | MCP 官方 Spec (modelcontextprotocol.io)、MCP 官方 Registry |
| **A2A 層 (L4)** | A2A Protocol Spec (a2a-protocol.org)、A2A GitHub、MCP vs A2A 工程指南 (workos.com) |
| **CLI Agent / Runtime 層** | Pi Framework (github.com/badlogic/pi-mono)、Claude Code docs |
| **研究資料** | ProtocolBench — protocol 選擇影響完成時間 36% (openreview.net) |
| **Agent Skills 標準** | agentskills.io |
| **Pi + Gemma 4 實戰** | Patrick Loeber 文章 (patloeber.com)、Pi README (GitHub)、Parallel.ai CLI agent 對照、HazAT Gemma 4 sub-agent 失敗模式實錄 |
| **Gemma 4 模型** | A Visual Guide to Gemma 4 (Maarten Grootendorst)、Google Developers: Gemma 4 on the Edge |