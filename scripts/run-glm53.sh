#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
profile=${1:-agents4}
split_mode=layer
spec_type=none
draft_max=1
if (( $# > 0 )); then
    shift
fi

case "$profile" in
    speed)
        profile_args=(-c 65536 -np 1 -ncmoe 29 -ts 36,11 -ub 128 --load-mode dio --no-mmproj-offload --ctx-checkpoints 4)
        template_args='{"reasoning_effort":"low","clear_thinking":true}'
        ;;
    tensor)
        split_mode=tensor
        profile_args=(-c 65536 -np 1 -ncmoe 31 -ts 1,1 -ub 256 --load-mode mmap --no-mmproj-offload --ctx-checkpoints 4)
        template_args='{"reasoning_effort":"low","clear_thinking":true}'
        ;;
    mtp)
        spec_type=draft-mtp
        profile_args=(-c 65536 -np 1 -ncmoe 30 -ts 37,10 -ub 128 --load-mode mmap --no-mmproj-offload --ctx-checkpoints 4)
        template_args='{"reasoning_effort":"low","clear_thinking":true}'
        ;;
    tensor-mtp)
        split_mode=tensor
        spec_type=draft-mtp
        draft_max=3
        profile_args=(-c 65536 -np 1 -ncmoe 33 -ts 1,1 -ub 256 --load-mode mmap --no-mmproj-offload --ctx-checkpoints 4)
        template_args='{"reasoning_effort":"low","clear_thinking":true}'
        ;;
    mtp-max)
        spec_type=draft-mtp
        profile_args=(-c 1048576 -np 1 -ncmoe 38 -ts 40,7 -ub 64 --load-mode mmap --no-mmproj-offload --ctx-checkpoints 2 --no-op-offload --override-tensor 'blk\.44\.ffn_gate_exps\.weight=CPU')
        template_args='{"reasoning_effort":"max","clear_thinking":true}'
        ;;
    max)
        profile_args=(-c 1048576 -np 1 -ncmoe 34 -ts 37,10 -ub 64 --load-mode mmap --no-mmproj-offload --ctx-checkpoints 2 --no-op-offload --override-tensor 'blk\.44\.ffn_gate_exps\.weight=CPU')
        template_args='{"reasoning_effort":"max","clear_thinking":true}'
        ;;
    agents2)
        profile_args=(-c 262144 -np 2 -ncmoe 31 -ts 37,10 -ub 128 --load-mode mmap --no-mmproj-offload --ctx-checkpoints 4 --kv-unified-per-slot 131072)
        template_args='{"reasoning_effort":"high","clear_thinking":true}'
        ;;
    agents4)
        profile_args=(-c 262144 -np 4 -ncmoe 32 -ts 37,10 -ub 256 --load-mode mmap --no-mmproj-offload --ctx-checkpoints 4 --kv-unified-per-slot 65536)
        template_args='{"reasoning_effort":"high","clear_thinking":true}'
        ;;
    mtp-agents4)
        spec_type=draft-mtp
        profile_args=(-c 262144 -np 4 -ncmoe 34 -ts 39,8 -ub 128 --load-mode mmap --no-mmproj-offload --ctx-checkpoints 4 --kv-unified-per-slot 65536)
        template_args='{"reasoning_effort":"high","clear_thinking":true}'
        ;;
    tensor-agents2)
        split_mode=tensor
        profile_args=(-c 262144 -np 2 -ncmoe 35 -ts 1,1 -ub 256 --load-mode mmap --no-mmproj-offload --ctx-checkpoints 4 --kv-unified-per-slot 131072)
        template_args='{"reasoning_effort":"high","clear_thinking":true}'
        ;;
    tensor-agents4)
        split_mode=tensor
        profile_args=(-c 262144 -np 4 -ncmoe 35 -ts 1,1 -ub 256 --load-mode mmap --no-mmproj-offload --ctx-checkpoints 4 --kv-unified-per-slot 65536)
        template_args='{"reasoning_effort":"high","clear_thinking":true}'
        ;;
    tensor-mtp-agents4)
        split_mode=tensor
        spec_type=draft-mtp
        profile_args=(-c 262144 -np 4 -ncmoe 40 -ts 1,1 -ub 128 --load-mode mmap --no-mmproj-offload --ctx-checkpoints 4 --kv-unified-per-slot 65536)
        template_args='{"reasoning_effort":"high","clear_thinking":true}'
        ;;
    *)
        printf 'Usage: %s {speed|tensor|mtp|tensor-mtp|mtp-max|max|agents2|agents4|mtp-agents4|tensor-agents2|tensor-agents4|tensor-mtp-agents4} [llama-server options]\n' "$0" >&2
        exit 2
        ;;
esac

if [[ "$split_mode" == tensor ]]; then
    ulimit -S -s 65536
fi

exec env -u LLAMA_ARG_SPEC_TYPE -u LLAMA_ARG_SPEC_DRAFT_MODEL "$repo_dir/build/bin/llama-server" \
    -m "$HOME/LLMs/GLM-5.3-Flash-UD-Q2_K_XL.gguf" \
    --mmproj "$HOME/LLMs/mmproj-BF16-GLM-5.3-Flash.gguf" \
    -ngl 99 -sm "$split_mode" --fit off \
    -kvu -ctk q8_0 -ctv q8_0 -fa on \
    -t 24 -tb 24 -b 2048 \
    --cache-ram 0 --no-cache-idle-slots \
    --temp 1.0 --top-p 0.95 --top-k 0 --min-p 0 \
    --jinja --spec-type "$spec_type" --spec-draft-n-max "$draft_max" --spec-draft-p-min 0.75 -ctkd q8_0 -ctvd q8_0 --chat-template-kwargs "$template_args" \
    --host 127.0.0.1 --port 8080 --metrics \
    "${profile_args[@]}" "$@"
