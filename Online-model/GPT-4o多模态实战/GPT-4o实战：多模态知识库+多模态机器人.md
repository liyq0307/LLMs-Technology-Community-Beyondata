```python
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY") #调用你的API keys
from openai import OpenAI #导入OpenAI
client = OpenAI() #与OpenAI产生链接
```

```python
response = client.chat.completions.create(
  model="gpt-4o",
  messages=[{"role": "user","content": "你好呀GPT！"}],
  max_tokens=300,
)
```

```python
print(response.choices[0].message.content)
```

```plaintext
你好！你今天过得怎么样？有什么我可以帮你的吗？
```

```python
# 使用GPT-4o描述图像
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "请描述以下图片的内容"},
            {"type": "image_url"
             , "image_url": {"url": 
                             "https://skojiangdoc.oss-cn-beijing.aliyuncs.com/LLM/Video/page1_img1.jpeg"}}
        ,{"type": "image_url"
             , "image_url": {"url":  #改掉下面这行 ↓
                             "https://skojiangdoc.oss-cn-beijing.aliyuncs.com/LLM/Video/page1_img1.jpeg"}}
        
        ]}
    ],
    max_tokens=300,
)

description = response.choices[0].message.content
print(description)
```

```plaintext
这张图片展示了一种办公场景：

- 中央是一台蓝色的电脑显示器，上面显示着图表和其他数据图形。
- 左侧是一位穿黄色上衣和黑色裤子的女性，她站在一个装有文件的橙色文件夹旁边，文件夹上有一些文件。
- 右侧是一位穿蓝色外套和黑裤子的男性，他靠近显示器，伸手指向显示器的内容。
- 背景中还有一些齿轮图案，象征着机械和系统的运转。
- 显示器旁边有一个绿色的植物。

这幅插画一般用于描述与数据分析、办公、团队协作相关的场景。
```

![](images/369126dc-b8a6-4ec8-a917-e3ce67fadd86.jpeg)

## 处理多模态数据

### 1. 使用PDF解析库提取文本和表格

```python
pdf_path = r"D:\pythonwork\LLM\Course\annual-and-transition-report-of-foreign-private-issuers-sections-13-or-15-d.pdf"
```

```python
import pdfplumber

def extract_text_and_tables_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        texts = []
        tables = []
        #取出从190-200页的信息
        #这里是核心财务信息
        for page in pdf.pages[190:200]: 
            texts.append(page.extract_text())
            tables.append(page.extract_tables())
    return texts, tables

texts, tables = extract_text_and_tables_from_pdf(pdf_path)
```

```python
texts[:200]
```

```plaintext
['Table of Contents\nBILIBILI INC.\nCONSOLIDATED BALANCE SHEETS (Continued)\n(All amounts in thousands, except for share data)\nDecember 31, December 31, December 31,\n2022 2023 2023\nRMB RMB US$\nNote 2(e)\nShareholders’ equity\nOrdinary shares:\nClass Y Ordinary Shares (US$0.0001 par value; 100,000,000 shares authorized, 83,715,114\nshares issued and outstanding as of December 31, 2022; US$0.0001 par value; 100,000,000\nshares authorized, 83,715,114 shares issued and outstanding as of December 31, 2023) 52 52 7\nClass Z Ordinary Shares (US$0.0001 par value; 9,800,000,000 shares authorized, 316,202,303\nshares issued, 310,864,471 shares outstanding as of December 31, 2022; 9,800,000,000\nshares authorized, 337,546,303 shares issued, 328,441,712 shares outstanding as of\nDecember 31, 2023) 201 213 30\nAdditional paid-in capital 36,623,161 40,445,175 5,696,584\nStatutory reserves 36,173 44,749 6,303\nAccumulated other comprehensive income 58,110 212,477 29,927\nAccumulated deficit (21,479,869) (26,310,766) (3,705,794)\nTotal Bilibili Inc.’s shareholders’ equity 15,237,828 14,391,900 2,027,057\nNoncontrolling interests 1,759 12,367 1,742\nTotal shareholders’ equity 15,239,587 14,404,267 2,028,799\nTotal liabilities and shareholders’ equity 41,830,570 33,159,067 4,670,357\nThe accompanying notes are an integral part of these consolidated financial statements.\nF-5',
 'Table of Contents\nBILIBILI INC.\nCONSOLIDATED STATEMENTS OF OPERATIONS AND COMPREHENSIVE LOSS\n(All amounts in thousands, except for share and per share data)\nFor the Year Ended December 31,\n2021 2022 2023 2023\nRMB RMB RMB US$\nNote 2(e)\nNet revenues 19,383,684 21,899,167 22,527,987 3,173,001\nCost of revenues (15,340,537 ) (18,049,872 ) (17,086,122 ) (2,406,530 )\nGross profit 4,043,147 3,849,295 5,441,865 766,471\nOperating expenses:\nSales and marketing expenses (5,794,853) (4,920,745) (3,916,150) (551,578)\nGeneral and administrative expenses (1,837,506) (2,521,134) (2,122,432) (298,938)\nResearch and development expenses (2,839,862) (4,765,360) (4,467,470) (629,230)\nTotal operating expenses (10,472,221 ) (12,207,239 ) (10,506,052 ) (1,479,746 )\nLoss from operations (6,429,074 ) (8,357,944 ) (5,064,187 ) (713,275 )\nOther (expense)/income:\nInvestment loss, net (including impairments) (194,183) (532,485) (435,644) (61,359)\nInterest income 70,367 281,051 542,472 76,406\nInterest expense (155,467) (250,923) (164,927) (23,229)\nExchange losses (15,504) (19,745) (35,575) (5,011)\nDebt extinguishment gain — 1,318,594 292,213 41,157\nOthers, net 10,411 157,944 132,640 18,682\nTotal other (expense)/income, net (284,376 ) 954,436 331,179 46,646\nLoss before income tax expenses (6,713,450 ) (7,403,508 ) (4,733,008 ) (666,629 )\nIncome tax (95,289) (104,145) (78,705) (11,085)\nNet loss (6,808,739 ) (7,507,653 ) (4,811,713 ) (677,714 )\nNet loss/(profit) attributable to noncontrolling interests 19,511 10,640 (10,608) (1,494)\nNet loss attributable to the Bilibili Inc.’s shareholders (6,789,228 ) (7,497,013 ) (4,822,321 ) (679,208 )\nNet loss (6,808,739 ) (7,507,653 ) (4,811,713 ) (677,714 )\nOther comprehensive (loss)/income:\nForeign currency translation adjustments (420,991) 337,972 154,367 21,742\nTotal other comprehensive (loss)/income (420,991) 337,972 154,367 21,742\nTotal comprehensive loss (7,229,730 ) (7,169,681 ) (4,657,346 ) (655,972 )\nComprehensive loss/(income) attributable to noncontrolling interests 19,511 10,640 (10,608 ) (1,494 )\nComprehensive loss attributable to the Bilibili Inc.’s shareholders (7,210,219 ) (7,159,041 ) (4,667,954 ) (657,466 )\nNet loss per share, basic (17.87 ) (18.99 ) (11.67 ) (1.64 )\nNet loss per share, diluted (17.87) (18.99) (11.67) (1.64)\nNet loss per ADS, basic (17.87) (18.99) (11.67) (1.64)\nNet loss per ADS, diluted (17.87) (18.99) (11.67) (1.64)\nWeighted average number of ordinary shares, basic 379,898,121 394,863,584 413,210,271 413,210,271\nWeighted average number of ordinary shares, diluted 379,898,121 394,863,584 413,210,271 413,210,271\nWeighted average number of ADS, basic 379,898,121 394,863,584 413,210,271 413,210,271\nWeighted average number of ADS, diluted 379,898,121 394,863,584 413,210,271 413,210,271\nShare-based compensation expenses included in:\nCost of revenues 76,232 69,096 63,724 8,975\nSales and marketing expenses 53,452 59,041 56,649 7,979\nGeneral and administrative expenses 553,526 554,976 596,950 84,079\nResearch and development expenses 316,607 357,570 415,321 58,497\nThe accompanying notes are an integral part of these consolidated financial statements.\nF-6',
 'Table of Contents\nBILIBILI INC.\nCONSOLIDATED STATEMENTS OF CHANGES IN SHAREHOLDERS’ EQUITY\n(All amounts in thousands, except for share data)\nOrdinary shares\nClass Y Ordinary Class Z Ordinary Additional Accumulated other Total\nShares Shares paid-in Statutory comprehensive Accumulated Noncontrolling sharehold ers’\nShares Amount Shares Amount capital reserves income/(loss) deficit interests equity\nRMB RMB RMB RMB RMB RMB RMB RMB\nBalance at December 31, 2020 83,715,114 52 268,204,838 172 14,616,302 17,884 141,129 (7,175,339) 182,004 7,782,204\nNet loss — — — — — — — (6,789,228) (19,511) (6,808,739)\nShare-based compensation — — — — 999,817 — — — — 999,817\nShare issuance from exercise of\nshare options — — 3,262,562 3 — — — — — 3\nShare issuance upon secondary\npublic offering (“HK IPO”), net\nof issuance costs of\nHKD337,143 — — 28,750,000 18 19,266,792 — — — — 19,266,810\nAcquisition of subsidiaries — — 2,056,825 1 632,747 — — — (14,749) 617,999\nIssuance of Class Z ordinary\nshares related to long-term\ninvestments — — 1,045,700 1 (1) — — — — —\nShare issuance upon the\nconversion of convertible senior\nnotes — — 2,854,277 3 449,908 — — — — 449,911\nCapital injection in subsidiaries by\nnoncontrolling interests — — — — — — — — 2,187 2,187\nAppropriation to statutory reserves — — — — — 6,737 — (6,737) — —\nPurchase of noncontrolling\ninterests — — 715,271 1 (35,604) — — — (137,532) (173,135)\nForeign currency translation\nadjustment — — — — — — (420,991) — — (420,991)\nBalance at December 31, 2021 83,715,114 52 306,889,473 199 35,929,961 24,621 (279,862 ) (13,971,304 ) 12,399 21,716,066\nThe accompanying notes are an integral part of these consolidated financial statements.\nF-7',
 'Table of Contents\nBILIBILI INC.\nCONSOLIDATED STATEMENTS OF CHANGES IN SHAREHOLDERS’ EQUITY (Continued)\n(All amounts in thousands, except for share data)\nOrdinary shares\nClass Y Ordinary Class Z Ordinary Additional Accumulated other Total\nShares Shares paid-in Statutory comprehensive Accumulated Noncontrolling sharehold ers’\nShares Amount Shares Amount capital reserves (loss)/income deficit interests equity\nRMB RMB RMB RMB RMB RMB RMB RMB\nBalance at December 31, 2021 83,715,114 52 306,889,473 199 35,929,961 24,621 (279,862) (13,971,304) 12,399 21,716,066\nNet loss — — — — — — — (7,497,013) (10,640) (7,507,653)\nShare-based compensation — — — — 1,040,683 — — — — 1,040,683\nShare issuance from exercise of\nshare options — — 3,929,433 4 — — — — — 4\nPurchase of noncontrolling\ninterests — — 45,000 * * — — — — —\nShare issuance upon the\nconversion of convertible\nsenior notes — — 565 * 96 — — — — 96\nRepurchase of shares — — (2,640,832) (2) (347,579) — — — — (347,581)\nAppropriation to statutory\nreserves — — — — — 11,552 — (11,552) — —\nForeign currency translation\nadjustments — — — — — — 337,972 — — 337,972\nBalance at December 31, 2022 83,715,114 52 308,223,639 201 36,623,161 36,173 58,110 (21,479,869 ) 1,759 15,239,587\n* Less than 1.\nThe accompanying notes are an integral part of these consolidated financial statements.\nF-8',
 'Table of Contents\nBILIBILI INC.\nCONSOLIDATED STATEMENTS OF CHANGES IN SHAREHOLDERS’ EQUITY (Continued)\n(All amounts in thousands, except for share data)\nOrdinary shares\nClass Y Ordinary Class Z Ordinary Additional Accumulated other Total\nShares Shares paid-in Statutory comprehensive Accumulated Noncontrolling sharehold ers’\nShares Amount Shares Amount capital reserves income deficit interests equity\nRMB RMB RMB RMB RMB RMB RMB RMB\nBalance at December 31, 2022 83,715,114 52 308,223,639 201 36,623,161 36,173 58,110 (21,479,869) 1,759 15,239,587\nNet (loss)/income — — — — — — — (4,822,321) 10,608 (4,811,713)\nShare-based compensation — — — — 1,132,644 — — — — 1,132,644\nIssuance of Class Z ordinary shares\nupon new ADS offering (“ADS\noffering”) — — 15,344,000 10 2,689,370 — — — — 2,689,380\nShare issuance from exercise of share\noptions — — 2,210,741 2 — — — — — 2\nShare issuance from vest of restricted\nshare units — — 22,500 * — — — — — *\nAppropriation to statutory reserves — — — — — 8,576 — (8,576) — —\nForeign currency translation\nadjustments — — — — — — 154,367 — — 154,367\nBalance at December 31, 2023 83,715,114 52 325,800,880 213 40,445,175 44,749 212,477 (26,310,766 ) 12,367 14,404,267\n* Less than 1.\nThe accompanying notes are an integral part of these consolidated financial statements.\nF-9',
 'Table of Contents\nBILIBILI INC.\nCONSOLIDATED STATEMENTS OF CASH FLOWS\n(All amounts in thousands)\nFor the Year Ended December 31,\n2021 2022 2023 2023\nRMB RMB RMB US$\nNote 2(e)\nCash flows from operating activities:\nNet loss (6,808,739) (7,507,653) (4,811,713) (677,714)\nAdjustments to reconcile net loss to net cash (used in)/provided by operating activities:\nDepreciation of property and equipment 538,601 755,452 727,193 102,423\nAmortization of intangible assets 1,903,226 2,581,325 2,003,199 282,145\nAmortization of right-of-use assets 161,873 229,295 169,294 23,845\nAmortization of debt issuance costs 25,234 48,167 23,299 3,282\nShare-based compensation expenses 999,817 1,040,683 1,132,644 159,530\nAllowance for/(reversal of) expected credit losses 189,165 130,549 (16,000) (2,254)\nInventory provision 24,454 191,088 10,431 1,469\nDeferred income taxes (21,492) (36,495) (25,376) (3,574)\nUnrealized exchange (gains)/losses (6,592) 2,277 931 131\nUnrealized fair value changes of investments 200,274 251,383 66,462 9,361\nLoss on disposal of property and equipment 611 748 646 91\nTermination expenses of certain game projects — 525,762 354,811 49,974\nGain from disposal of subsidiaries and long-term investments (4,413) (178,378) (3,857) (543)\nLoss from equity method investments 37,179 211,620 112,142 15,795\nRevaluation of previously held equity interests 31,462 (152,153) 86,680 12,209\nImpairments of long-term investments 91,493 465,645 278,891 39,281\nGain of convertible senior notes repurchase — (1,318,594) (292,213) (41,157)\nChanges in operating assets and liabilities:\nAccounts receivable (429,460) (59,866) (262,215) (36,932)\nAmount due from related parties 8,792 (74,117) 75,273 10,602\nPrepayments and other assets (1,747,744) (556,214) 342,537 48,245\nOther long-term assets (138,396) (444,326) 185,783 26,167\nAccounts payable 1,056,847 46,862 (60,458) (8,517)\nSalary and welfare payable 254,213 396,113 (182,171) (25,658)\nTaxes payable 77,365 130,684 29,006 4,085\nDeferred revenue 494,551 173,935 133,902 18,859\nAccrued liabilities and other payables 319,702 (1,026,319) 341,102 48,042\nAmount due to related parties — (21,210) (12,007) (1,691)\nOther long-term liabilities 94,969 282,367 (141,594) (19,943)\nNet cash (used in)/provided by operating activities (2,647,008 ) (3,911,370 ) 266,622 37,553\nCash flows from investing activities:\nPurchase of property and equipment (965,410) (760,427) (181,897) (25,620)\nPurchase of intangible assets (2,721,799) (1,977,897) (1,148,247) (161,727)\nPurchase of short-term investments (71,748,847) (70,578,711) (13,547,650) (1,908,147)\nMaturities of short-term investments 60,524,888 81,698,532 16,328,255 2,299,787\nCash consideration paid for purchase of subsidiaries, net of cash acquired (521,984) (1,179,764) (70,000) (9,859)\nCash paid for long-term investments including loans (6,716,491) (1,466,311) (132,602) (18,677)\nRepayment of loans from investees 539,225 596,766 684,834 96,457\nCash received from disposal/return of long-term assets 74,604 612,214 100,574 14,166\nPlacements of time deposits (10,697,444) (10,245,026) (9,961,925) (1,403,108)\nMaturities of time deposits 7,655,147 13,909,967 9,690,806 1,364,921\nImpact to cash resulting from deconsolidation of subsidiaries — (125) — —\nNet cash (used in)/provided by investing activities (24,578,111 ) 10,609,218 1,762,148 248,193\nCash flows from financing activities:\nProceeds of short-term loan 1,332,597 1,701,532 1,950,482 274,720\nRepayment of short-term loan (214,882) (1,450,627) (2,032,295) (286,243)\nPurchase of noncontrolling interests (104,696) (56,741) (7,027) (990)\nCapital injections from noncontrolling interests 2,187 — — —\nProceeds from exercise of employees’ share options 3 4 2 *\nProceeds from issuance of ordinary shares net of issuance costs 19,288,423 — 2,689,380 378,792\nRepurchase of shares — (347,581) — —\nProceeds from/(Repurchase of) issuance of convertible senior notes, net of issuance costs of US$23,402, nil and\nnil, respectively 10,085,520 (4,201,506) (7,675,227) (1,081,033)\nNet cash provided by/(used in) financing activities 30,389,152 (4,354,919 ) (5,074,685 ) (714,754 )\nEffect of exchange rate changes on cash and cash equivalents and restricted cash held in foreign currencies (319,034 ) 321,350 100,349 14,134\nNet increase/(decrease) in cash and cash equivalents and restricted cash 2,844,999 2,664,279 (2,945,566) (414,874)\nCash and cash equivalents and restricted cash at beginning of the year 4,678,109 7,523,108 10,187,387 1,434,863\nCash and cash equivalents and restricted cash at end of the year 7,523,108 10,187,387 7,241,821 1,019,989\nIncluding:\nF-10',
 'Table of Contents\nBILIBILI INC.\nCONSOLIDATED STATEMENTS OF CASH FLOWS (Continued)\n(All amounts in thousands)\nCash and cash equivalents at end of the year 7,523,108 10,172,584 7,191,821 1,012,947\nRestricted cash at end of the year — 14,803 50,000 7,042\nSupplemental disclosures of cash flows information:\nCash paid for income taxes, net of tax refund 73,717 80,591 103,391 14,562\nCash paid for interest expense 116,226 200,172 167,291 23,562\nSupplemental schedule of non-cash investing and financing activities:\nProperty and equipment purchases financed by accounts payable 183,203 46,066 65,377 9,208\nAcquisitions and investments financed by/(reduction of) payables 731,503 202,018 (80,000) (11,268)\nIntangible assets purchases financed by payables 830,596 824,929 783,828 110,400\nIssuance of ordinary shares in the business combination, purchase of noncontrolling interests and\ninvestment addition 1,207,980 — — —\nIssuance of ordinary shares in connection with debt conversion 449,914 96 — —\n* Less than 1.\nThe accompanying notes are an integral part of these consolidated financial statements.\nF-11',
 'Table of Contents\nBILIBILI INC.\nNOTES TO THE CONSOLIDATED FINANCIAL STATEMENTS\n1. Operations\nBilibili Inc. (the “Company” or “Bilibili”) is an iconic brand and a leading video community for young generations in China. Incorporated as a\nlimited liability company in the Cayman Islands in December 2013, the Company, through its consolidated subsidiaries, variable interest entities\n(“VIEs”) and subsidiaries of the VIEs (collectively referred to as the “Group”), is primarily engaged in the operation of providing online entertainment\nservices to users in the People’s Republic of China (the “PRC” or “China”).\nIn April 2018, the Company completed its IPO on the NASDAQ Global Select Market. In March 2021, the Company successfully listed its\nClass Z ordinary shares on the main board of the Hong Kong Stock Exchange. The Company issued a total 28,750,000 Class Z ordinary shares in the\nglobal offering, including the fully exercised over-allotment option of 3,750,000 Class Z ordinary shares. Net proceeds from the global offering,\nincluding the over-allotment option, after deducting underwriting fees and other offering expenses, were approximately HKD22.9 billion (RMB19.3\nbillion).\nOn October 3, 2022 (the “Primary Conversion Effective Date”), the Company’s voluntary conversion of its secondary listing status to primary\nlisting on the main board of the Hong Kong Stock Exchange became effective. The Company became a dual-primary listed company on the main board\nof Hong Kong Stock Exchange in Hong Kong and the NASDAQ Global Select Market in the United States.\nIn January 2023, the Company completed the offering of 15,344,000 ADSs at US$26.65 per ADS. The amount of net proceeds from such offering\n(after deducting all applicable costs and expenses including but not limited to selling commission) is approximately US$396.9 million (RMB2,689.4\nmillion). Shortly thereafter, the Company completed repurchased of an aggregate principal amount of US$384.8 million (RMB2.6 billion) of its\noutstanding 0.5% convertible senior notes due December 2026 with an aggregate purchase price of US$331.2 million (RMB2.2 billion), which was\nfunded by the net proceeds from the ADS Offering.\nAs of December 31, 2023, the Company’s major subsidiaries, VIEs and subsidiaries of the VIEs are as follows:\nPercentage of\nDirect or Indire ct\nPlace and Year of Economic\nMajor Subsidiaries Incorporation Ownership Principal Activities\nBilibili HK Limited Hong Kong, 2014 100 Investment holding\nHode HK Limited Hong Kong, 2014 100 Investment holding\nChaodian HK Limited Hong Kong, 2019 100 Investment holding\nBilibili Co., Ltd. Japan, 2014 100 Business development\nHode Shanghai Limited (“Hode Shanghai”) PRC, 2014 100 Technology development1\nShanghai Bilibili Technology Co., Ltd. PRC, 2016 100 Technology development1\nChaodian (Shanghai) Technology Co., Ltd. PRC, 2019 100 E-commerce and advertising1\nF-12',
 'Table of Contents\nBILIBILI INC.\nNOTES TO THE CONSOLIDATED FINANCIAL STATEMENTS (Continued)\n1. Operations (Continued)\nPercentage of\nDirect or\nIndirect\nPlace and Year of Economic\nMajor VIEs and VIEs’ subsidiaries Incorporation Interest Principal Activities\nShanghai Hode Information Technology Co., Ltd. (“Hode Information\nTechnology”) PRC, 2013 100* Mobile game operation2\nShanghai Kuanyu Digital Technology Co., Ltd. (“Shanghai Kuanyu”) PRC, 2014 100* Video distribution and game distribution2\nSharejoy Network Technology Co., Ltd. (“Sharejoy Network”) PRC, 2014 100* Game distribution2\nShanghai Hehehe Culture Communication Co., Ltd. (“Shanghai Hehehe”) PRC, 2014 100* Comics distribution2\nShanghai Anime Tamashi Cultural Media Co., Ltd. (“Shanghai Anime\nTamashi”) PRC, 2015 100* E-commerce platform2\n* Hode Shanghai is the primary beneficiary of the major VIEs and VIEs’ subsidiaries.\n1 These companies were established in the PRC in the form of wholly foreign-owned enterprises.\n2 These companies were established in the PRC in the form of investment solely by legal corporations or controlled by natural person(s).\nF-13',
 'Table of Contents\nBILIBILI INC.\nNOTES TO THE CONSOLIDATED FINANCIAL STATEMENTS (Continued)\n1. Operations (Continued)\nContractual agreements with major VIEs\nIn order to comply with the PRC laws and regulations which prohibit or restrict foreign control of companies involved in provision of internet\ncontent services, the Group operates its restricted businesses in the PRC through the VIEs, whose equity interests are held by certain founders of the\nGroup. The Company obtained control over these VIEs by entering into a series of contractual arrangements with the legal shareholders who are also\nreferred to as nominee shareholders. These nominee shareholders are the legal owners of the VIEs. However, the rights of those nominee shareholders\nhave been transferred to the Company through the contractual arrangements.\nThe contractual arrangements that are used to control the VIEs include powers of attorney, exclusive technology consulting and services\nagreements or exclusive business cooperation agreements, equity pledge agreements and exclusive option agreements. Management concluded that the\nCompany, through the contractual arrangements, has the power to direct the activities that most significantly impact the VIEs’ economic performance,\nbears the risks of and enjoys the rewards normally associated with ownership of the VIEs, and therefore the Company is the ultimate primary\nbeneficiary of these VIEs. As such, the Company consolidates the financial statements of these VIEs. Consequently, the financial results of the VIEs\nwere included in the Group’s consolidated financial statements in accordance with the presentation as stated in Note 2(a).\nThe following is a summary of the contractual agreements entered into by and among the Company’s relevant subsidiaries, the VIEs, and\nrespective nominee shareholders of the VIEs.\nExclusive Technology Consulting and Services Agreements. Under the exclusive technology consulting and services agreements between the\nCompany’s relevant subsidiaries and the VIEs, the Company’s relevant subsidiaries have the exclusive right to provide the VIEs consulting and services\nrelated to, among other things, research and development, system operation, advertising, internal training and technical support. The Company’s relevant\nsubsidiaries have the exclusive ownership of intellectual property rights created as a result of the performance of these agreements. These VIEs shall pay\nthe Company’s relevant subsidiaries an annual service fee, which are subject to the adjustment by the Company’s relevant subsidiaries at its sole\ndiscretion. These agreements will remain effective for a 10 year’s term and then be automatically renewed, unless the Company’s relevant subsidiaries\ngive the VIEs a termination notice 90 days before the term ends. On December 23, 2020, the above agreements were replaced by the exclusive business\ncooperation agreements, which contain terms substantially similar to the exclusive business cooperation agreements described above, the exclusive\nbusiness cooperation agreements have an infinite period commencing from December 23, 2020, unless the Company’s relevant subsidiaries give the\nVIEs a termination notice 30 days before the term ends.\nF-14']
```

```python
count = 0
for i in texts:
    count+= i.__len__()
```

```python
count #21w字
```

```plaintext
21759
```

```python
tables[:200]
```

```plaintext
[[[['Shareholders’ equity', '', '', '', '', '', '', '', '', '']],
  [['Class Y Ordinary Shares (US$0.0001 par value; 100,000,000 shares authorized, 83,715,114\nshares issued and outstanding as of December 31, 2022; US$0.0001 par value; 100,000,000\nshares authorized, 83,715,114 shares issued and outstanding as of December 31, 2023)',
    '',
    '52',
    '',
    '',
    '52',
    '',
    '',
    '7',
    '']],
  [['Additional paid-in capital',
    '',
    '36,623,161',
    '',
    '',
    '40,445,175',
    '',
    '',
    '5,696,584',
    '']],
  [['Accumulated other comprehensive income',
    '',
    '58,110',
    '',
    '',
    '212,477',
    '',
    '',
    '29,927',
    '']],
  [['Total Bilibili Inc.’s shareholders’ equity',
    '',
    '15,237,828',
    '',
    '',
    '14,391,900',
    '',
    '',
    '2,027,057',
    '']],
  [['Total shareholders’ equity',
    '',
    '15,239,587',
    '',
    '',
    '14,404,267',
    '',
    '',
    '2,028,799',
    '']]],
 [[['Net revenues',
    '',
    '19,383,684',
    '',
    '',
    '21,899,167',
    '',
    '',
    '22,527,987',
    '',
    '',
    '3,173,001',
    '']],
  [['Gross profit',
    '',
    '4,043,147',
    '',
    '',
    '3,849,295',
    '',
    '',
    '5,441,865',
    '',
    '',
    '766,471',
    '']],
  [['Sales and marketing expenses',
    '',
    '(5,794,853',
    ')',
    '',
    '(4,920,745',
    ')',
    '',
    '(3,916,150',
    ')',
    '',
    '(551,578',
    ')']],
  [['Research and development expenses',
    '',
    '(2,839,862',
    ')',
    '',
    '(4,765,360',
    ')',
    '',
    '(4,467,470',
    ')',
    '',
    '(629,230',
    ')']],
  [['Loss from operations',
    '',
    '(6,429,074',
    ')',
    '',
    '(8,357,944',
    ')',
    '',
    '(5,064,187',
    ')',
    '',
    '(713,275',
    ')']],
  [['Investment loss, net (including impairments)',
    '',
    '(194,183',
    ')',
    '',
    '(532,485',
    ')',
    '',
    '(435,644',
    ')',
    '',
    '(61,359',
    ')']],
  [['Interest expense',
    '',
    '(155,467',
    ')',
    '',
    '(250,923',
    ')',
    '',
    '(164,927',
    ')',
    '',
    '(23,229',
    ')']],
  [['Debt extinguishment gain',
    '',
    '—',
    '',
    '',
    '1,318,594',
    '',
    '',
    '292,213',
    '',
    '',
    '41,157',
    '']],
  [['Total other (expense)/income, net',
    '',
    '(284,376',
    ')',
    '',
    '954,436',
    '',
    '',
    '331,179',
    '',
    '',
    '46,646',
    '']],
  [['Income tax',
    '',
    '(95,289',
    ')',
    '',
    '(104,145',
    ')',
    '',
    '(78,705',
    ')',
    '',
    '(11,085',
    ')']],
  [['Net loss/(profit) attributable to noncontrolling interests',
    '',
    '19,511',
    '',
    '',
    '10,640',
    '',
    '',
    '(10,608',
    ')',
    '',
    '(1,494',
    ')']],
  [['Net loss',
    '',
    '(6,808,739',
    ')',
    '',
    '(7,507,653',
    ')',
    '',
    '(4,811,713',
    ')',
    '',
    '(677,714',
    ')']],
  [['Foreign currency translation adjustments',
    '',
    '(420,991',
    ')',
    '',
    '337,972',
    '',
    '',
    '154,367',
    '',
    '',
    '21,742',
    '']],
  [['Total comprehensive loss',
    '',
    '(7,229,730',
    ')',
    '',
    '(7,169,681',
    ')',
    '',
    '(4,657,346',
    ')',
    '',
    '(655,972',
    ')']],
  [['Comprehensive loss attributable to the Bilibili Inc.’s shareholders',
    '',
    '(7,210,219',
    ')',
    '',
    '(7,159,041',
    ')',
    '',
    '(4,667,954',
    ')',
    '',
    '(657,466',
    ')']],
  [['Net loss per share, diluted',
    '',
    '(17.87',
    ')',
    '',
    '(18.99',
    ')',
    '',
    '(11.67',
    ')',
    '',
    '(1.64',
    ')']],
  [['Net loss per ADS, diluted',
    '',
    '(17.87',
    ')',
    '',
    '(18.99',
    ')',
    '',
    '(11.67',
    ')',
    '',
    '(1.64',
    ')']],
  [['Weighted average number of ordinary shares, diluted',
    '',
    '379,898,121',
    '',
    '',
    '394,863,584',
    '',
    '',
    '413,210,271',
    '',
    '',
    '413,210,271',
    '']],
  [['Weighted average number of ADS, diluted',
    '',
    '379,898,121',
    '',
    '',
    '394,863,584',
    '',
    '',
    '413,210,271',
    '',
    '',
    '413,210,271',
    '']],
  [['Cost of revenues',
    '',
    '76,232',
    '',
    '',
    '69,096',
    '',
    '',
    '63,724',
    '',
    '',
    '8,975',
    '']],
  [['General and administrative expenses',
    '',
    '553,526',
    '',
    '',
    '554,976',
    '',
    '',
    '596,950',
    '',
    '',
    '84,079',
    '']]],
 [[['Balance at December 31, 2020',
    '',
    '83,715,114',
    '',
    '52',
    '',
    '268,204,838',
    '',
    '172',
    '',
    '14,616,302',
    '',
    '',
    '17,884',
    '',
    '',
    '141,129',
    '',
    '',
    '(7,175,339',
    ')',
    '',
    '182,004',
    '',
    '',
    '7,782,204',
    '']],
  [['Share-based compensation',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '999,817',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '999,817',
    '']],
  [['Share issuance upon secondary\npublic offering (“HK IPO”), net\nof issuance costs of\nHKD337,143',
    '',
    '—',
    '',
    '—',
    '',
    '28,750,000',
    '',
    '18',
    '',
    '19,266,792',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '19,266,810',
    '']],
  [['Issuance of Class Z ordinary\nshares related to long-term\ninvestments',
    '',
    '—',
    '',
    '—',
    '',
    '1,045,700',
    '',
    '1',
    '',
    '(1',
    ')',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '']],
  [['Capital injection in subsidiaries by\nnoncontrolling interests',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '2,187',
    '',
    '',
    '2,187',
    '']],
  [['Purchase of noncontrolling\ninterests',
    '',
    '—',
    '',
    '—',
    '',
    '715,271',
    '',
    '1',
    '',
    '(35,604',
    ')',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '(137,532',
    ')',
    '',
    '(173,135',
    ')']],
  [['Balance at December 31, 2021',
    '',
    '83,715,114',
    '',
    '52',
    '',
    '306,889,473',
    '',
    '199',
    '',
    '35,929,961',
    '',
    '',
    '24,621',
    '',
    '',
    '(279,862',
    ')',
    '',
    '(13,971,304',
    ')',
    '',
    '12,399',
    '',
    '',
    '21,716,066',
    '']]],
 [[['Balance at December 31, 2021',
    '',
    '83,715,114',
    '',
    '52',
    '',
    '306,889,473',
    '',
    '',
    '199',
    '',
    '',
    '35,929,961',
    '',
    '',
    '24,621',
    '',
    '',
    '(279,862',
    ')',
    '',
    '(13,971,304',
    ')',
    '',
    '12,399',
    '',
    '',
    '21,716,066',
    '']],
  [['Share-based compensation',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '1,040,683',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '1,040,683',
    '']],
  [['Purchase of noncontrolling\ninterests',
    '',
    '—',
    '',
    '—',
    '',
    '45,000',
    '',
    '',
    '*',
    '',
    '',
    '*',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '']],
  [['Repurchase of shares',
    '',
    '—',
    '',
    '—',
    '',
    '(2,640,832',
    ')',
    '',
    '(2',
    ')',
    '',
    '(347,579',
    ')',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '(347,581',
    ')']],
  [['Foreign currency translation\nadjustments',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '337,972',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '337,972',
    '']]],
 [[['Balance at December 31, 2022',
    '',
    '83,715,114',
    '',
    '52',
    '',
    '308,223,639',
    '',
    '201',
    '',
    '36,623,161',
    '',
    '36,173',
    '',
    '',
    '58,110',
    '',
    '(21,479,869',
    ')',
    '',
    '',
    '1,759',
    '',
    '15,239,587',
    ''],
   ['Net (loss)/income',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '(4,822,321',
    ')',
    '',
    '',
    '10,608',
    '',
    '(4,811,713',
    ')'],
   ['Share-based compensation',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '1,132,644',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '—',
    '',
    '',
    '',
    '—',
    '',
    '1,132,644',
    ''],
   ['Issuance of Class Z ordinary shares\nupon new ADS offering (“ADS\noffering”)',
    '',
    '—',
    '',
    '—',
    '',
    '15,344,000',
    '',
    '10',
    '',
    '2,689,370',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '—',
    '',
    '',
    '',
    '—',
    '',
    '2,689,380',
    ''],
   ['Share issuance from exercise of share\noptions',
    '',
    '—',
    '',
    '—',
    '',
    '2,210,741',
    '',
    '2',
    '',
    '—',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '—',
    '',
    '',
    '',
    '—',
    '',
    '2',
    ''],
   ['Share issuance from vest of restricted\nshare units',
    '',
    '—',
    '',
    '—',
    '',
    '22,500',
    '',
    '*',
    '',
    '—',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '—',
    '',
    '',
    '',
    '—',
    '',
    '*',
    ''],
   ['Appropriation to statutory reserves',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '8,576',
    '',
    '',
    '—',
    '',
    '(8,576',
    ')',
    '',
    '',
    '—',
    '',
    '—',
    ''],
   ['Foreign currency translation\nadjustments',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '—',
    '',
    '',
    '154,367',
    '',
    '—',
    '',
    '',
    '',
    '—',
    '',
    '154,367',
    ''],
   ['Balance at December 31, 2023',
    '',
    '83,715,114',
    '',
    '52',
    '',
    '325,800,880',
    '',
    '213',
    '',
    '40,445,175',
    '',
    '44,749',
    '',
    '',
    '212,477',
    '',
    '(26,310,766',
    ')',
    '',
    '',
    '12,367',
    '',
    '14,404,267',
    '']]],
 [[['Cash flows from operating activities:',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '']],
  [['Adjustments to reconcile net loss to net cash (used in)/provided by operating activities:',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '']],
  [['Amortization of intangible assets',
    '',
    '1,903,226',
    '',
    '',
    '2,581,325',
    '',
    '',
    '2,003,199',
    '',
    '',
    '282,145',
    '']],
  [['Amortization of debt issuance costs',
    '',
    '25,234',
    '',
    '',
    '48,167',
    '',
    '',
    '23,299',
    '',
    '',
    '3,282',
    '']],
  [['Allowance for/(reversal of) expected credit losses',
    '',
    '189,165',
    '',
    '',
    '130,549',
    '',
    '',
    '(16,000',
    ')',
    '',
    '(2,254',
    ')']],
  [['Deferred income taxes',
    '',
    '(21,492',
    ')',
    '',
    '(36,495',
    ')',
    '',
    '(25,376',
    ')',
    '',
    '(3,574',
    ')']],
  [['Unrealized fair value changes of investments',
    '',
    '200,274',
    '',
    '',
    '251,383',
    '',
    '',
    '66,462',
    '',
    '',
    '9,361',
    '']],
  [['Termination expenses of certain game projects',
    '',
    '—',
    '',
    '',
    '525,762',
    '',
    '',
    '354,811',
    '',
    '',
    '49,974',
    '']],
  [['Loss from equity method investments',
    '',
    '37,179',
    '',
    '',
    '211,620',
    '',
    '',
    '112,142',
    '',
    '',
    '15,795',
    '']],
  [['Impairments of long-term investments',
    '',
    '91,493',
    '',
    '',
    '465,645',
    '',
    '',
    '278,891',
    '',
    '',
    '39,281',
    '']],
  [['Changes in operating assets and liabilities:',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '']],
  [['Amount due from related parties',
    '',
    '8,792',
    '',
    '',
    '(74,117',
    ')',
    '',
    '75,273',
    '',
    '',
    '10,602',
    '']],
  [['Other long-term assets',
    '',
    '(138,396',
    ')',
    '',
    '(444,326',
    ')',
    '',
    '185,783',
    '',
    '',
    '26,167',
    '']],
  [['Salary and welfare payable',
    '',
    '254,213',
    '',
    '',
    '396,113',
    '',
    '',
    '(182,171',
    ')',
    '',
    '(25,658',
    ')']],
  [['Deferred revenue',
    '',
    '494,551',
    '',
    '',
    '173,935',
    '',
    '',
    '133,902',
    '',
    '',
    '18,859',
    '']],
  [['Amount due to related parties',
    '',
    '—',
    '',
    '',
    '(21,210',
    ')',
    '',
    '(12,007',
    ')',
    '',
    '(1,691',
    ')']],
  [['Net cash (used in)/provided by operating activities',
    '',
    '(2,647,008',
    ')',
    '',
    '(3,911,370',
    ')',
    '',
    '266,622',
    '',
    '',
    '37,553',
    '']],
  [['Purchase of property and equipment',
    '',
    '(965,410',
    ')',
    '',
    '(760,427',
    ')',
    '',
    '(181,897',
    ')',
    '',
    '(25,620',
    ')']],
  [['Purchase of short-term investments',
    '',
    '(71,748,847',
    ')',
    '',
    '(70,578,711',
    ')',
    '',
    '(13,547,650',
    ')',
    '',
    '(1,908,147',
    ')']],
  [['Cash consideration paid for purchase of subsidiaries, net of cash acquired',
    '',
    '(521,984',
    ')',
    '',
    '(1,179,764',
    ')',
    '',
    '(70,000',
    ')',
    '',
    '(9,859',
    ')']],
  [['Repayment of loans from investees',
    '',
    '539,225',
    '',
    '',
    '596,766',
    '',
    '',
    '684,834',
    '',
    '',
    '96,457',
    '']],
  [['Placements of time deposits',
    '',
    '(10,697,444',
    ')',
    '',
    '(10,245,026',
    ')',
    '',
    '(9,961,925',
    ')',
    '',
    '(1,403,108',
    ')']],
  [['Impact to cash resulting from deconsolidation of subsidiaries',
    '',
    '—',
    '',
    '',
    '(125',
    ')',
    '',
    '—',
    '',
    '',
    '—',
    '']],
  [['Cash flows from financing activities:',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '']],
  [['Repayment of short-term loan',
    '',
    '(214,882',
    ')',
    '',
    '(1,450,627',
    ')',
    '',
    '(2,032,295',
    ')',
    '',
    '(286,243',
    ')']],
  [['Capital injections from noncontrolling interests',
    '',
    '2,187',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '',
    '',
    '—',
    '']],
  [['Proceeds from issuance of ordinary shares net of issuance costs',
    '',
    '19,288,423',
    '',
    '',
    '—',
    '',
    '',
    '2,689,380',
    '',
    '',
    '378,792',
    '']],
  [['Proceeds from/(Repurchase of) issuance of convertible senior notes, net of issuance costs of US$23,402, nil and\nnil, respectively',
    '',
    '10,085,520',
    '',
    '',
    '(4,201,506',
    ')',
    '',
    '(7,675,227',
    ')',
    '',
    '(1,081,033',
    ')']],
  [['Effect of exchange rate changes on cash and cash equivalents and restricted cash held in foreign currencies',
    '',
    '(319,034',
    ')',
    '',
    '321,350',
    '',
    '',
    '100,349',
    '',
    '',
    '14,134',
    '']],
  [['Cash and cash equivalents and restricted cash at beginning of the year',
    '',
    '4,678,109',
    '',
    '',
    '7,523,108',
    '',
    '',
    '10,187,387',
    '',
    '',
    '1,434,863',
    '']],
  [['Including:', '', '', '', '', '', '', '', '', '', '', '', '']]],
 [[['Cash and cash equivalents at end of the year',
    '',
    '7,523,108',
    '',
    '10,172,584',
    '',
    '7,191,821',
    '',
    '',
    '1,012,947',
    '']],
  [['Supplemental disclosures of cash flows information:',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '']],
  [['Cash paid for interest expense',
    '',
    '116,226',
    '',
    '200,172',
    '',
    '167,291',
    '',
    '',
    '23,562',
    '']],
  [['Property and equipment purchases financed by accounts payable',
    '',
    '183,203',
    '',
    '46,066',
    '',
    '65,377',
    '',
    '',
    '9,208',
    '']],
  [['Intangible assets purchases financed by payables',
    '',
    '830,596',
    '',
    '824,929',
    '',
    '783,828',
    '',
    '',
    '110,400',
    '']],
  [['Issuance of ordinary shares in connection with debt conversion',
    '',
    '449,914',
    '',
    '96',
    '',
    '—',
    '',
    '',
    '—',
    '']]],
 [[['Bilibili HK Limited',
    '',
    'Hong Kong, 2014',
    '',
    '100',
    '',
    'Investment holding']],
  [['Chaodian HK Limited',
    '',
    'Hong Kong, 2019',
    '',
    '100',
    '',
    'Investment holding']],
  [['Hode Shanghai Limited (“Hode Shanghai”)',
    '',
    'PRC, 2014',
    '',
    '100',
    '',
    'Technology development1']],
  [['Chaodian (Shanghai) Technology Co., Ltd.',
    '',
    'PRC, 2019',
    '',
    '100',
    '',
    'E-commerce and advertising1']]],
 [[['Shanghai Hode Information Technology Co., Ltd. (“Hode Information\nTechnology”)',
    '',
    'PRC, 2013',
    '',
    '100*',
    '',
    'Mobile game operation2']],
  [['Sharejoy Network Technology Co., Ltd. (“Sharejoy Network”)',
    '',
    'PRC, 2014',
    '',
    '100*',
    '',
    'Game distribution2']],
  [['Shanghai Anime Tamashi Cultural Media Co., Ltd. (“Shanghai Anime\nTamashi”)',
    '',
    'PRC, 2015',
    '',
    '100*',
    '',
    'E-commerce platform2']]],
 [[[None,
    'In order to comply with the PRC laws and regulations which prohibit or restrict foreign control of companies involved in provision of internet'],
   ['content services, the Group operates its restricted businesses in the PRC through the VIEs, whose equity interests are held by certain founders of the',
    None],
   ['Group. The Company obtained control over these VIEs by entering into a series of contractual arrangements with the legal shareholders who are also',
    None],
   ['referred to as nominee shareholders. These nominee shareholders are the legal owners of the VIEs. However, the rights of those nominee shareholders',
    None],
   ['have been transferred to the Company through the contractual arrangements.',
    None]]]]
```

```python
#转换成字符串
texts = repr(texts)
tables = repr(tables)
```

### 2. 提取pdf中的图像

```python
import fitz  # PyMuPDF
import os

def extract_images_from_pdf(pdf_path, start_page, end_page, output_dir):
    # 确认输出路径是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_document = fitz.open(pdf_path)
    images = []

    for page_num in range(start_page - 1, end_page):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)

        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]

            # 获取图片的元数据
            image_name = f"page{page_num + 1}_img{image_index + 1}"
            image_ext = base_image["ext"]
            images.append((image_name, image_ext, image_bytes))
    
    return images

# 页数范围
pdf_path = r"D:\pythonwork\LLM\Course\哔哩哔哩(B站)用户人群画像分析报告2023.pdf"
start_page = 1
end_page = 10
output_dir = r"D:\pythonwork\LLM\Course\output_images_folder"

# 提取图片
extracted_images = extract_images_from_pdf(pdf_path, start_page, end_page, output_dir)

# 保存图片到本地
for image_name, image_ext, image_bytes in extracted_images:
    image_path = os.path.join(output_dir, f"{image_name}.{image_ext}")
    with open(image_path, "wb") as image_file:
        image_file.write(image_bytes)
        print(f"Saved {image_path}")
```

```plaintext
Saved D:\pythonwork\LLM\Course\output_images_folder\page1_img1.jpeg
Saved D:\pythonwork\LLM\Course\output_images_folder\page2_img1.jpeg
Saved D:\pythonwork\LLM\Course\output_images_folder\page3_img1.jpeg
Saved D:\pythonwork\LLM\Course\output_images_folder\page4_img1.jpeg
Saved D:\pythonwork\LLM\Course\output_images_folder\page5_img1.jpeg
Saved D:\pythonwork\LLM\Course\output_images_folder\page6_img1.jpeg
Saved D:\pythonwork\LLM\Course\output_images_folder\page7_img1.jpeg
Saved D:\pythonwork\LLM\Course\output_images_folder\page8_img1.jpeg
Saved D:\pythonwork\LLM\Course\output_images_folder\page9_img1.jpeg
```

### 3.【选做】使用高精度OCR提取图像中的文本

```python
"""
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter

def preprocess_image(image):
    # 转换为灰度图像
    image = image.convert('L')
    # 增强对比度
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    # 应用滤镜
    image = image.filter(ImageFilter.MedianFilter())
    return image

def extract_text_from_image(image):
    preprocessed_image = preprocess_image(image)
    text = pytesseract.image_to_string(preprocessed_image)
    return text

# 指定需要处理的页数范围
start_page = 190
end_page = 200

# 加载指定页数的PDF页面
images = convert_from_path(pdf_path)
selected_images = images[start_page - 1:end_page]

# 提取文本
ocr_texts = [extract_text_from_image(image) for image in selected_images]

# 输出提取的文本
for page_num, text in enumerate(ocr_texts, start=start_page):
    print(f"Page {page_num}:\n{text}\n")
"""
```

### 4.上传图像并生成URL集合

```python
import oss2
import os

# 阿里云AccessKey ID和AccessKey Secret
access_key_id = 'your_access_key_id'
access_key_secret = 'your_access_key_secret'
bucket_name = 'your_bucket_name'
endpoint = 'your_endpoint'  # 例如: 'oss-cn-hangzhou.aliyuncs.com'

# 初始化Bucket
auth = oss2.Auth(access_key_id, access_key_secret)
bucket = oss2.Bucket(auth, endpoint, bucket_name)

def upload_image(file_path, object_name):
    """
    上传单个图像到OSS
    :param file_path: 本地文件路径
    :param object_name: 上传到OSS后的对象名
    :return: 上传的图像URL
    """
    bucket.put_object_from_file(object_name, file_path)
    url = f"https://{bucket_name}.{endpoint}/{object_name}"
    return url

def batch_upload_images(local_dir):
    """
    批量上传图像并返回URL列表
    :param local_dir: 本地目录路径
    :return: 上传的图像URL列表
    """
    image_urls = []
    for file_name in os.listdir(local_dir):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(local_dir, file_name)
            object_name = f'images/{file_name}'  # 在OSS中的存储路径
            url = upload_image(file_path, object_name)
            image_urls.append(url)
    return image_urls

# 本地图像目录
local_directory = 'path/to/your/images'

# 批量上传图像并获取URL
uploaded_image_urls = batch_upload_images(local_directory)

# 打印所有上传的图像URL
for url in uploaded_image_urls:
    print(url)
```

```python
image_urls = ["https://skojiangdoc.oss-cn-beijing.aliyuncs.com/LLM/Video/page1_img1.jpeg"
             ,"https://skojiangdoc.oss-cn-beijing.aliyuncs.com/LLM/Video/page2_img1.jpeg"
             ,"https://skojiangdoc.oss-cn-beijing.aliyuncs.com/LLM/Video/page3_img1.jpeg"
             ,"https://skojiangdoc.oss-cn-beijing.aliyuncs.com/LLM/Video/page4_img1.jpeg"
             ,"https://skojiangdoc.oss-cn-beijing.aliyuncs.com/LLM/Video/page5_img1.jpeg"
             ,"https://skojiangdoc.oss-cn-beijing.aliyuncs.com/LLM/Video/page6_img1.jpeg"
             ,"https://skojiangdoc.oss-cn-beijing.aliyuncs.com/LLM/Video/page7_img1.jpeg"
             ,"https://skojiangdoc.oss-cn-beijing.aliyuncs.com/LLM/Video/page8_img1.jpeg"
             ,"https://skojiangdoc.oss-cn-beijing.aliyuncs.com/LLM/Video/page9_img1.jpeg"]
```

### 5.将多模态信息传给GPT-4o

```python
# 组合所有提取的内容
ocr_texts = ""
combined_texts = "\n".join(texts + tables + ocr_texts)
messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "请用300字总结一下现在我给你的全部信息，包括图像"},
        {"type": "text", "text": combined_texts},
        *[{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
    ]}
]

response = client.chat.completions.create(
  model="gpt-4o",
  messages=messages,
  max_tokens=300,
)

# 打印响应
description = response.choices[0].message.content
print(description)
```

```plaintext
收到的信息包括表格、财务数据以及图表。这些数据和图表主要涵盖了B站2023年第三季度财报的内容，突出显示了关键指标，如日均活跃用户、总营收及毛利率等。2023年第三季度，B站日均活跃用户达1.03亿，较去年同比增长14%；月均活跃用户为3.41亿。总营收为58.1亿元人民币，毛利率上升至25.0%。其中，增值服务业务收入为26.0亿元，同比增长17%；广告业务收入为16.4亿元，同比增长21%；游戏业务收入为9.9亿元。调整后的净亏损同比大幅收窄51%，实现季度环比五连涨。UP主活跃同比增长21%，月均投稿量增加37%，并且“正式会员”保留率在第12个月达到约80%。此外，B站的广告收入高速增长，增收及减亏成效显著，形成了一个良好的财务表现。插图生动地展示了B站的财报数据及趋势，其中包括一些图形和图表来表示关键数据指标及其同比变化。总体而言，B站在第三季度取得了显著的用户增长及收入表现，如图所示，其各项关键指标均呈现上涨趋势。
```

```python
#与gpt-4o进行对话

messages.append({"role":"assistant","content":response.choices[0].message.content})

prompt = "请帮我整理出过去2年中哩哔哩公司的发展情况"

messages.append({"role":"user","content":prompt})

response = client.chat.completions.create(
  model="gpt-4o",
  messages=messages,
  max_tokens=1000,
)

print(response.choices[0].message.content)
```

```plaintext
过去两年中，哔哩哔哩公司（B站）的发展情况可从以下几个方面进行总结：

### 1. 用户增长
- **日均活跃用户（DAU）：**
  B站的日均活跃用户持续增长，从2022年Q1的约79.4百万增长至2023年Q3的1.03亿，增长率显著。
- **月均活跃用户（MAU）：**
  月均活跃用户数量也增加，2023年Q3达到了3.41亿。
- **用户活跃度：**
  用户日均使用时长创历史新高，达到100分钟。

### 2. 财务表现
- **总营收：**
  总收入逐渐增加，2023年Q3的营收为58.1亿元人民币，比去年同期增长。
- **毛利率：**
  毛利率连续五个季度环比上升，2023年Q3毛利率为25%，比2022年Q2的15%有所增长。
- **业务收入：**
  - **增值服务业务**：增加到26.0亿元，同比增长17%。
  - **广告业务**：收入提升至16.4亿元，同比增长21%。
  - **游戏业务**：收入达到9.9亿元。
- **调整后的净亏损：**
  2023年Q3调整后的净亏损同比减少51%。

### 3. 内容及平台生态
- **UP主（内容创作者）增长：**
  活跃UP主同比增长21%，投稿量增长37%。
- **内容多样性：**
  专业用户自制内容（PUGV）覆盖多个有趣且有用的领域，如职场知识、家居、养育、科技数码、汽车等。
  
### 4. 会员及互动
- **正式会员：**
  正式会员数量持续增加，正式会员第12个月的保留率约为80%，说明平台的用户粘性较高。
- **月均互动数：**
  2023年Q3的平均互动数达170亿，同比增长18%。

### 5. 重大事件
- **股票上市及募集资金：**
  2021年3月，B站在香港成功挂牌上市，发行了28,750,000股Class Z普通股份，募集到约19.3亿港元（约合22.9亿人民币）。2023年初，再次通过发行ADS募集资金，进一步支持公司的发展。

### 总结：
哔哩哔哩在过去两年中，用户活跃度显著增加，财务表现持续改善，营收和毛利率逐步提升。同时，B站在内容丰富性和内容质量方面不断提升，吸引了大量内容创作者和用户。凭借稳健的用户增长策略和多样的内容生态，B站在互联网视频社区领域保持持续发展的势头。
```

```python
#messages.append({"role":"assistant","content":response.choices[0].message.content})

prompt = "请问过去几年中，哔哩哔哩的利润状况如何？告诉我毛利率的具体数据即可"

messages.append({"role":"user","content":prompt})

response = client.chat.completions.create(
  model="gpt-4o",
  messages=messages,
  max_tokens=1000,
)

description = response.choices[0].message.content

print(description)
```

```plaintext
根据你提供的信息，以下是哔哩哔哩在过去几年的毛利率数据：

- **2022年Q2：** 15.0%
- **2022年Q3：** 18.2%
- **2022年Q4：** 20.3%
- **2023年Q1：** 21.8%
- **2023年Q2：** 23.1%

从这些数据可以看出，哔哩哔哩的毛利率在过去几年中呈现出逐步增长的趋势。
```

```python
# 组合所有提取的内容
ocr_texts = ""
combined_texts = "\n".join(texts + tables + ocr_texts)
messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "请用300字总结一下现在我给你的全部信息，包括图像"},
        {"type": "text", "text": combined_texts},
        *[{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
    ]}
]

response = client.chat.completions.create(
  model="gpt-4o",
  messages=messages,
  max_tokens=300,
)

# 打印响应
description = response.choices[0].message.content
```

```python
response = client.images.generate(
  model="dall-e-3",
  prompt= description + "根据我所提供的数据，绘制横坐标为时间、纵坐标为百分比的折线图",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url

# 获取生成的图像URL
print(response.data[0].url)
```

```plaintext
https://oaidalleapiprodscus.blob.core.windows.net/private/org-HSwQITpPfCnOXCUCglzxrecL/user-GMg4tqRRi7GglFyhJY97QbH7/img-Pu3VxHTw4Y4srobRnhs6NAvL.png?st=2024-05-14T11%3A23%3A22Z&se=2024-05-14T13%3A23%3A22Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2024-05-13T15%3A49%3A02Z&ske=2024-05-14T15%3A49%3A02Z&sks=b&skv=2021-08-06&sig=5IuA2Oje0qRFN%2B4e/75b1v1K/uWZCM6NDoQEqG/3V0s%3D
```

![](images/default.png)

```python
```



**更多大模型技术内容学习+加入大模型技术社群**

**扫码添加助理英英，回复“大模型”，了解更多大模型技术详情哦👇**

![](images/f339b04b7b20233dd1509c7fb36d5c0.png)

此外，**扫码回复“LLM”**，即可领取**公开课课件、代码、数据**等\~
