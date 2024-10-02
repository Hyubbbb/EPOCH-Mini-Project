# 🔥 EPOCH Datathon
**이 레포지토리는 데이터 사이언스 연합동아리 EPOCH 멤버의 Mini Project를 기록하기 위해 개설되었습니다.**

> **진행 기간**: `2024.10.01~2024.10.06`


## Datathon 개요
> **본 Datathon은 동일한 데이터셋을 기반으로 각 팀별로 각자 주제를 선정 및 데이터 분석을 통하여 최종 결과를 도출합니다.**

## Dataset
> 본 Datathon에서 사용될 데이터셋은 총 **5개의 CSV 파일**로 구성되어 있습니다. 
> 
> 데이터셋은 이 레포지토리에 업로드되어 있으며, 각 팀은 이를 기반으로 분석을 진행합니다.

### 파일 목록:
1. `credits.csv`: Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object.
2. `keywords.csv`: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object.
3. `links.csv`: The file that contains the TMDB and IMDB IDs of all the movies featured in the Full MovieLens dataset.
4. `movies_metadata.csv`: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.
5. `ratings.csv`: Ratings data

## 제출 가이드라인
> **공통 경로**: `EPOCH_Datathon/Mini_Project_Submissions/`
### 예선
각 팀은 **코드**, **보드**를 제출해야 하며, 각 파일 양식은 아래와 같습니다.
- **코드**: `Preliminary_Round/TeamN/1st_miniproj_code_팀명.ipynb`
- **보드**: `Preliminary_Round/TeamN/1st_miniproj_board_팀명.png`
  - 주의: 템플릿으로 제공되는 `pptx` 확장자가 아닌 `png` 확장자로 제출
  - 템플릿: `1st_miniproj_board_팀명.pptx`

### 본선
- **코드**: `Final_Round/TeamN/1st_miniproj_code_팀명.ipynb`
- **발표 자료**: `Final_Round/TeamN/1st_miniproj_ppt_팀명.pptx`

### 폴더 구조:
```
EPOCH_Datathon/
├── data
└── Mini_Project_Submissions/
    ├── Preliminary_Round/
    │   ├── Team1/
    │   ├── .../
    │   └── Team9/
    └──  Final_Round/
        ├── Team1/
        ├── .../
        └── Team4/

```

### 🗓️ 제출 일정
- **예선 제출 마감**: `2024-10-06 06:00:00`
  - 제출 파일: 코드(`ipynb`), 보드(`png`)
- **본선 제출 마감**: `2024-10-08 19:00:00`
  - 제출 파일: 최종 코드(`ipynb`), PPT(`pptx`)

## 💯 평가 기준
### 1️⃣ 예선 평가 기준
- **주제 선정의 참신함** - `17.5%`
- **해결 방식의 논리성 및 정확성** - `17.5%`
- **결과물의 완성도** - `17.5%`
- **3분 PR의 전달성** - `17.5%`
- **주관 평가** - `30%`

### 2️⃣ 본선 평가 기준
- **동료 평가(예선 점수)** - `50%`
- **전문가 평가** - `30%`
- **발표 자료 및 발표 평가** - `20%`

---