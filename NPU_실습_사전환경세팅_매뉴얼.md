# NPU 실습 사전 환경 세팅 매뉴얼

> FuriosaAI NPU(RNGD) 원격 서버에 접속해 실습 환경(Miniconda 가상환경)을 세팅하기까지의 전체 절차. Windows 로컬 PC에서 원격 Linux 서버(NPU)에 SSH로 접속하는 구조다.

---

## ⚠️ 보안 주의 (먼저 읽기)

이 문서를 **다른 사람과 공유하거나 깃허브·노션 등 외부에 올릴 때는** 아래 값들을 절대 넣지 말 것. 본문은 전부 플레이스홀더(`<...>`)로 작성돼 있고, 실제 값은 **본인 PC에서만** 아래 표에 채워 쓴다.

| 플레이스홀더 | 뜻 | 민감도 |
|---|---|---|
| `<개인키파일>` | 발급받은 개인 SSH 키 파일 (예: id_rsa_11) | **최상** — 본인 전용, 유출 시 서버 탈취 가능. 파일 내용·파일명 모두 공유 금지 |
| `<NPU_HOST>` | NPU 서버 주소 (예: eval2.nxt.furiosa.ai) | 높음 — 외부 공개 시 무단 접속 시도 대상 |
| `<SSH_PORT>` | 개인별로 발급된 SSH 포트 번호 | 높음 |
| `<WINDOWS_USER>` | Windows 사용자 이름 | 중간 |
| `<WSL_USER>` / 비밀번호 | WSL Ubuntu 계정·비밀번호 | 높음 — 비밀번호는 문서에 적지 말 것 |

> 개인 SSH 키는 교육기관에서 1인 1키로 발급된다. **키 파일은 메신저·메일·클라우드에 올리지 말고, 로컬의 `.ssh` 폴더에만 두고 권한을 잠근다**(아래 절차).

---

## 0. 전체 흐름

### 한 문장 요약

> 내 **윈도우 PC에서 SSH로 원격 Furiosa 서버(리눅스 + NPU)에 접속** → 그 서버에 이미 깔린 **미니콘다 가상환경을 켜서** → 그 서버의 **NPU로 실습**한다. (WSL은 접속을 위한 도구일 뿐, 안 써도 된다.)

### 구조 그림

```
┌─────────────────────────┐          ┌──────────────────────────────────────┐
│   내 윈도우 PC (로컬)     │          │   원격 Furiosa 서버 (리눅스, root)      │
│                         │          │                                        │
│  SSH로 접속하는 "출발점"  │  ──SSH──▶ │   ★ 미니콘다가 여기 깔려 있음           │
│   · 방법 A: 윈도우 기본   │  (키로    │   ★ NPU 하드웨어가 여기 달려 있음       │
│   · 방법 B: WSL(Ubuntu)  │   인증)   │   → 접속하는 순간 NPU 서버 안이다        │
│                         │          │                                        │
│  (WSL = 내 PC 안의 리눅스,│          │  실습 = 이 서버에서 conda 환경 켜고 진행 │
│   원격이 아니라 접속 도구) │          │                                        │
└─────────────────────────┘          └──────────────────────────────────────┘
        여기서 하는 일                          여기서 하는 일
   ① 접속 환경 준비                        ③ NPU 상태 확인 (furiosa-smi)
     (SSH 키 배치·권한·config/WSL)          ④ Miniconda 설치 + 가상환경
   ② ssh 로 서버에 접속                        (★ 서버는 공용 1대라 한 번 깔면 끝)
```

**헷갈리기 쉬운 점 2가지**
- **WSL은 "원격"이 아니다.** 내 윈도우 PC 안에 깐 리눅스로, 그냥 SSH 접속을 위한 출발점(도구)이다. 진짜 원격은 Furiosa 서버다. (WSL 없이 윈도우 기본 SSH로 바로 접속해도 된다 — 방법 A)
- **미니콘다·NPU는 내 PC가 아니라 원격 서버에 있다.** 그래서 접속 = NPU 서버 진입이고, 미니콘다 설치·확인도 전부 접속 후 서버 안에서 한다. 서버는 모두가 쓰는 공용 1대라, 한 번 설치해두면 집·학원 어디서 접속해도 그대로 있다.

접속 방법은 두 가지다. 둘 중 **하나만** 하면 된다.
- **방법 A — Windows에서 바로 접속**: Windows 기본 SSH 사용 (config + 키 권한 설정)
- **방법 B — WSL(Ubuntu) 통해 접속**: Windows 안에 리눅스(WSL)를 깔고 그 안에서 접속

접속에 성공한 뒤의 ③·④(NPU 확인, 콘다 설치)는 두 방법 모두 동일하다.

---

## 방법 A — Windows에서 바로 SSH 접속

### A-1. SSH config 작성

접속 정보를 매번 길게 입력하지 않도록 `config` 파일에 별칭(`furiosa-npu`)으로 등록한다.

- 파일 위치: `C:\Users\<WINDOWS_USER>\.ssh\config` (확장자 없는 `config` 파일)

```sshconfig
Host furiosa-npu
    HostName <NPU_HOST>
    User root
    Port <SSH_PORT>
    IdentityFile C:/Users/<WINDOWS_USER>/.ssh/<개인키파일>
    IdentitiesOnly yes
```

- **Host**: 내가 쓸 별칭. 이후 `ssh furiosa-npu`로 접속된다.
- **HostName / Port**: 서버 주소와 개인 발급 포트.
- **User**: 원격 서버 로그인 계정(`root`).
- **IdentityFile**: 사용할 개인키 경로(경로 구분자는 `/` 사용).
- **IdentitiesOnly yes**: 지정한 이 키만 사용(다른 키로 시도하지 않음).

### A-2. 개인키 권한 설정 (PowerShell)

SSH는 개인키 파일 권한이 느슨하면 접속을 거부한다. **소유자만 읽기 가능**하도록 잠근다.

```powershell
# 키 파일이 있는지 확인
Test-Path "$env:USERPROFILE\.ssh\<개인키파일>"

# 키 경로를 변수에 저장
$key = "$env:USERPROFILE\.ssh\<개인키파일>"

# 상속된 권한 제거 (물려받은 권한 전부 끊기)
icacls $key /inheritance:r

# 현재 사용자에게만 읽기(R) 권한 부여
icacls $key /grant:r "$($env:USERNAME):(R)"

# 적용된 권한 확인
icacls $key
```

- `icacls`는 Windows의 파일 권한 설정 명령어다. `/inheritance:r`로 상위 폴더에서 물려받은 권한을 모두 끊고, `/grant:r`로 **나 자신에게만** 읽기 권한을 준다. 이렇게 해야 SSH가 "키가 너무 공개돼 있다"며 막지 않는다.

### A-3. 접속

```powershell
ssh furiosa-npu
```

처음 접속 시 호스트를 신뢰할지 묻는다(`Are you sure you want to continue connecting?`) → `yes` 입력. 접속되면 프롬프트가 원격 서버(`root@...`)로 바뀐다. → **③(NPU 확인)** 으로 이동.

---

## 방법 B — WSL(Ubuntu)로 SSH 접속

### B-1. WSL에 Ubuntu 설치 (PowerShell)

```powershell
# 설치된 WSL 배포판·버전 확인
wsl -l -v

# Ubuntu 22.04 를 원하는 이름의 배포판으로 설치
wsl --install -d Ubuntu-22.04 --name <배포판이름>
```

- 최초 실행 시 **Linux 계정 이름과 비밀번호**를 만든다(Windows 계정과 별개). ⚠️ 비밀번호는 이 문서에 적지 말 것.
- 설치 후 실행: `wsl -d <배포판이름>`

### B-2. SSH 키 배치 & 권한 (WSL Ubuntu 안에서)

```bash
cd ~                      # 홈 디렉터리로
sudo apt update           # 패키지 목록 갱신
mkdir .ssh                # SSH 키 보관 폴더 생성
cd .ssh
explorer.exe .            # 이 폴더를 Windows 탐색기로 열기 → 발급받은 <개인키파일> 복사해 넣기

ls -a                     # 키 파일이 들어왔는지 확인
chmod 600 <개인키파일>      # 소유자만 읽기/쓰기 가능하도록 권한 설정
```

- **`chmod 600`** 이 방법 A의 `icacls`에 해당한다. 리눅스에서 개인키 권한을 소유자 전용으로 잠그는 표준 방법이다.

### B-3. 접속 (WSL Ubuntu 안에서)

```bash
cd ~
ssh -i .ssh/<개인키파일> -p <SSH_PORT> root@<NPU_HOST>
```

- `-i`: 사용할 개인키 경로, `-p`: 포트 번호. (방법 A처럼 config를 만들면 `ssh furiosa-npu`로 줄일 수 있다.)
- 처음 접속 시 `yes`. 접속되면 프롬프트가 `root@...`로 바뀐다. → **③** 으로 이동.

---

## ③ NPU 상태 확인 (원격 NPU 서버에서)

접속에 성공하면 이제부터의 명령어는 **원격 Furiosa NPU 서버**에서 실행된다. `furiosa-smi`로 NPU가 정상인지 확인한다.

```bash
furiosa-smi            # 사용 가능한 하위 명령어 목록
furiosa-smi status     # NPU 동작 상태·메모리·코어 사용률
furiosa-smi info       # 펌웨어 버전·온도·전력·PCI 주소
furiosa-smi ps         # 현재 NPU 사용 중인 프로세스 (비어 있으면 없음)

watch -n 1 furiosa-smi status   # 1초마다 실시간 모니터링 (종료: Ctrl + C)
```

서버 자원도 확인할 수 있다.
```bash
lscpu      # CPU 사양 (아키텍처·코어·캐시 등)
free -h    # RAM / Swap 사용량 (읽기 쉬운 단위로)
```

- `furiosa-smi status`에서 `Liveness: alive`로 나오면 NPU가 정상 동작 중이다. 작업은 가급적 `root` 홈 디렉터리(`/root`, 프롬프트의 `~`)에서 한다.

---

## ④ Miniconda 설치 & 가상환경 (원격 NPU 서버에서)

실습용 파이썬 환경을 격리해서 쓰기 위해 Miniconda를 설치하고 가상환경을 만든다.
(공식 문서: https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install )

### ④-1. 설치

```bash
# 설치 스크립트 내려받기
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 설치 실행 (약관 동의 → 경로 확인 → 진행)
bash Miniconda3-latest-Linux-x86_64.sh

# conda 명령을 bash에서 쓰도록 초기화
~/miniconda3/bin/conda init bash

# 설정 즉시 반영
source ~/.bashrc
```

### ④-2. 가상환경 생성·활성화

```bash
# 파이썬 3.11 가상환경 생성 (-y: 확인 질문 자동 yes)
conda create -n <환경이름> python=3.11 -y

# 가상환경 활성화
conda activate <환경이름>

# 제대로 잡혔는지 확인
python --version       # Python 3.11.x
which python           # 경로에 miniconda3/envs/<환경이름> 포함되면 정상
python -m pip --version
```

### ④-3. 약관(TOS) 동의

일부 채널은 처음 사용 시 약관 동의가 필요하다. 설치·설치 중 오류가 나면 아래를 실행한다.
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

---

## 트러블슈팅 — conda 명령이 안 먹힐 때

`conda: command not found` 등으로 conda 경로가 안 잡히면, 설치 경로를 직접 확인하고 수동으로 초기화한다. (서버 계정이 root면 경로는 `/root/miniconda3`)

```bash
# conda 실행 파일이 실제로 있는지 확인
ls -l /root/miniconda3/bin/conda

# 전체 경로로 직접 실행해 환경 목록 확인
/root/miniconda3/bin/conda env list

# conda 초기화 스크립트를 현재 셸에 불러오기 (이러면 conda 명령 사용 가능)
source /root/miniconda3/etc/profile.d/conda.sh

conda env list
conda activate <환경이름>

# 셸을 새로 열어도 되게 영구 초기화
/root/miniconda3/bin/conda init bash
source ~/.bashrc
```

- 핵심은 `source .../conda.sh`로 현재 셸에 conda를 불러오거나, `conda init bash` + `source ~/.bashrc`로 영구 등록하는 것이다. 새 터미널을 열었을 때 conda가 안 보이면 초기화(`init`)가 안 된 상태다.

---

## 요약 체크리스트

- [ ] 개인 SSH 키(`<개인키파일>`)를 `.ssh` 폴더에 두고 권한 잠금 (Windows `icacls` / WSL `chmod 600`)
- [ ] (방법 A) `config` 작성 → `ssh furiosa-npu` / (방법 B) WSL에서 `ssh -i ... -p ...`
- [ ] 접속 후 `furiosa-smi status`로 NPU `alive` 확인
- [ ] Miniconda 설치 → `conda init bash` → `source ~/.bashrc`
- [ ] `conda create -n <환경이름> python=3.11 -y` → `conda activate <환경이름>` → `python --version` 확인
