# 4-Agent
4기 5주차 Agent

## 실행/설치 방법 (Conda)

이 폴더(`4-week5/`) 기준으로 **Conda 환경을 만들고**, `requirements.txt`로 의존성을 한 번에 설치합니다.

### 1) Conda 환경 생성/활성화

```bash
cd 4-week5
conda create -n agent python=3.11 -y
conda activate agent
```

### 2) 의존성 설치

```bash
pip install -r requirements.txt
```

### 3) .env 설정
노션의 사전 준비를 따라서
.env 파일을 설정해줍니다.


## 제출 방법

1. 자신의 깃허브 아이디로 브랜치 만든 후 hw_000.ipynb 파일명을 hw_이름으로 변경해주세요.
2. 작업 하시고 커밋 후 오리진으로 푸시해주시면 됩니다.
3. main 으로 풀리퀘 날려주시면 됩니다. merge 는 하시면 안돼요! 풀리퀘 제목 : 4기_5주차_[이름]

### 예시

```bash
git clone git@github.com:HateSlop/4-week5.git  # 클론
cd 4-week5 # 프로젝트 루트로 이동
git checkout -b '본인 브랜치명' # 브랜치 생성 (본인의 브랜치, 폴더 등 생성)
mkdir '본인 폴더명' # 개인 폴더 만들기
cd '본인 폴더명' # 개인 폴더로 이동
# 작업을 진행해주세요
git add . # 작업 후 add
git commit -m "[feat] ~~" # 커밋
git push origin '본인 브랜치명' # 오리진에 푸시
```

### 폴더구조

```bash
.
├── README.md       # 프로젝트 설명 리드미 (수정X)
├── agent.ipynb # 실습파일
└── ASSIGNMENT/    
    └── (개인 작업물) 
```

## 커밋 컨벤션

feat: 새로운 기능 추가  
fix: 버그 수정  
docs: 문서 수정  
style: 코드 포맷팅, 세미콜론 누락, 코드 변경이 없는 경우  
refactor: 코드 리팩토링  
test: 테스트 코드, 리팩토링 테스트 코드 추가  
chore: 빌드 업무 수정, 패키지 매니저 수정