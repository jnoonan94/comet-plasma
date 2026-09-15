# Git & Git Clone — Step-by-Step Guide
### Cometary Plasma Modeling Project

---

## What is Git?

Git is a **version control system** — a tool that tracks changes to files over time, lets you collaborate with others, and allows you to revert to any previous state of your project. It is the most widely used tool of its kind in scientific software and general software development.

A **repository** (repo) is just a project folder that Git is tracking. When you *clone* a repo, you download a full copy of it (including its entire history) to your local machine.

---

## Step 1 — Check if Git is Already Installed

Open a terminal and type:

```bash
git --version
```

If you see something like `git version 2.x.x`, Git is already installed and you can skip to Step 3.

If you get `command not found`, continue to Step 2.

---

## Step 2 — Install Git

**macOS (recommended method):**

Install via Homebrew (if you have it):
```bash
brew install git
```

Or install Xcode Command Line Tools, which bundles git:
```bash
xcode-select --install
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install git
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install git
```

After installation, confirm it worked:
```bash
git --version
```

---

## Step 3 — Configure Git (First-Time Setup)

Before using git, tell it who you are. This information is attached to any commits you make:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@auburn.edu"
```

You only need to do this once per machine.

---

## Step 4 — Understand What `git clone` Does

`git clone` copies a remote repository (hosted on GitHub, GitLab, Zenodo, etc.) to your local machine. The command structure is:

```bash
git clone <URL> [optional-folder-name]
```

- `<URL>` — the address of the remote repository
- `[optional-folder-name]` — if omitted, git uses the repo's name as the folder name

---

## Step 5 — Clone the `comet-plasma` Repository

### If the repo is hosted on GitHub:

Navigate to where you want to store the project, then clone it:

```bash
# Move to your Projects directory (or wherever you keep research code)
cd ~/Projects

# Clone the repo
git clone https://github.com/<username>/comet-plasma.git
```

Replace `<username>` with the GitHub username or organization that hosts the repo. Git will create a new folder called `comet-plasma/` and download everything into it.

### If you are cloning the Menura source from Zenodo:

The Menura v1 source is archived at: https://zenodo.org/records/6517018

Download the source archive from that page (Zenodo does not support direct `git clone`, so you'll download a `.zip` or `.tar.gz`). Then unzip it into your project folder:

```bash
cd ~/Projects/comet-plasma
unzip menura-v1.zip -d menura_source/
```

---

## Step 6 — Navigate the Cloned Repository

Once cloned, move into the folder:

```bash
cd comet-plasma
```

List its contents to get oriented:

```bash
ls -la
```

You should see folders like `menura_source/` and `tutorials/`, plus hidden git metadata (`.git/`).

Check the repository's history to see past commits:

```bash
git log --oneline
```

---

## Step 7 — Staying Up to Date

If the remote repository gets updated (e.g., new tutorials or fixes are pushed), you can pull those changes into your local copy:

```bash
# Make sure you're inside the repo folder
cd ~/Projects/comet-plasma

# Fetch and merge any new changes
git pull
```

---

## Quick Reference

| Command | What it does |
|---|---|
| `git --version` | Check if git is installed |
| `git config --global user.name "..."` | Set your name for commits |
| `git config --global user.email "..."` | Set your email for commits |
| `git clone <URL>` | Clone a remote repository |
| `cd <folder>` | Navigate into the cloned repo |
| `git log --oneline` | View the commit history |
| `git pull` | Pull the latest changes from remote |
| `git status` | See what files have changed locally |

---

## Troubleshooting

**"Permission denied (publickey)"**
This means GitHub requires SSH authentication instead of HTTPS. Either:
- Use the HTTPS URL (starts with `https://`) instead of the SSH URL (starts with `git@github.com:`)
- Or set up an SSH key by following: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

**"fatal: destination path already exists"**
A folder with that name already exists. Either delete it or clone into a different directory:
```bash
git clone <URL> comet-plasma-new
```

**"not a git repository"**
You're running a git command outside of a git repo folder. Make sure you `cd` into the cloned folder first.

---

*Guide written for the Cometary Plasma Modeling project — Auburn University Comet Group*
