#!/usr/bin/env bash
#
# Publish this site to GitHub Pages.
#
#   ./deploy.sh                review changes, then commit and push
#   ./deploy.sh "message"      same, with a commit message
#   ./deploy.sh -y             skip the confirmation prompt
#   ./deploy.sh -n             preview only, change nothing
#   ./deploy.sh --no-wait      push without waiting for the Pages build
#
set -euo pipefail

BRANCH="main"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dry_run=0
wait_for_build=1
assume_yes=0
message=""
for arg in "$@"; do
  case "$arg" in
    -n|--dry-run) dry_run=1 ;;
    -y|--yes)     assume_yes=1 ;;
    --no-wait)    wait_for_build=0 ;;
    -h|--help)    sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)           echo "unknown option: $arg" >&2; exit 64 ;;
    *)            message="$arg" ;;
  esac
done

git rev-parse --git-dir >/dev/null 2>&1 || { echo "not a git repository" >&2; exit 1; }

on="$(git symbolic-ref --short HEAD)"
[[ $on == "$BRANCH" ]] || { echo "on branch '$on', but Pages publishes '$BRANCH'" >&2; exit 1; }

pending="$(git status --porcelain)"

if ((dry_run)); then
  echo "== preview, nothing will be written =="
  if [[ -n $pending ]]; then
    echo "would commit:"
    git status --short
  else
    echo "no uncommitted changes"
  fi
  echo "would push $(git rev-list --count "@{upstream}..HEAD" 2>/dev/null || echo '?') unpushed commit(s)"
  exit 0
fi

if [[ -n $pending ]]; then
  # Everything on disk goes live, so show it before publishing a half-finished
  # edit. Run with -y once the diff is known to be good.
  if ((!assume_yes)); then
    echo "these changes will go live:"
    git status --short
    git diff --stat
    read -r -p "continue? [y/N] " reply
    [[ $reply == [yY] ]] || { echo "aborted"; exit 1; }
  fi
  git add -A
  git commit -q -m "${message:-Update site $(date '+%Y-%m-%d %H:%M')}"
  echo "committed  $(git log -1 --format='%h %s')"
else
  echo "no uncommitted changes"
fi

if [[ "$(git rev-list --count "@{upstream}..HEAD" 2>/dev/null || echo 1)" == 0 ]]; then
  echo "already up to date with origin/$BRANCH"
  exit 0
fi

git push -q origin "$BRANCH"
echo "pushed to origin/$BRANCH"

((wait_for_build)) || exit 0

if ! command -v gh >/dev/null; then
  echo "gh not installed, skipping build check"
  exit 0
fi

slug="$(git remote get-url origin | sed -E 's#.*github\.com[:/]##; s#\.git$##')"
sha="$(git rev-parse HEAD)"

# Poll the build for this exact commit; a stale "built" from the previous
# deploy would otherwise look like success immediately after pushing.
printf "building"
for _ in $(seq 1 60); do
  read -r state built_sha <<<"$(gh api "repos/$slug/pages/builds/latest" \
    --jq '"\(.status) \(.commit)"' 2>/dev/null || true)"
  if [[ ${built_sha:-} == "$sha" ]]; then
    case "${state:-}" in
      built)
        echo " ok"
        echo "live at $(gh api "repos/$slug/pages" --jq '.html_url' 2>/dev/null || echo "https://$slug")"
        exit 0
        ;;
      errored)
        echo " failed"
        gh api "repos/$slug/pages/builds/latest" --jq '.error.message' >&2
        exit 1
        ;;
    esac
  fi
  printf "."
  sleep 5
done

echo " timed out"
echo "check https://github.com/$slug/deployments"
