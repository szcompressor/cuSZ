# meta-doc: how to document cuSZ (incomplete)

## changelog update
Unlike most of documentations for cuSZ, changelog will non-styled text.

## code testing like a user

For a quick checking if cuSZ works, only two commands are needed; for zip,
```
./bin/cusz -f32 -m r2r -e 1e-4 -i ~/Parihaka_PSTM_far_stack.f32 -D parihaka -z
```
```
./bin/cusz -i ~/Parihaka_PSTM_far_stack.f32.cusza -x --origin ~/Parihaka_PSTM_far_stack.f32 --skip write.x
```

By using `--skip write.x`, we can still check compresion quality without output touching filesystem (to save time).

In addition, `--verbose` or `-V` (uppercase) to give machine information can be helpful.


## internal test from the team

The testing framework is not yet set up.

## subsequent log as an issue for a solved subproblem (v0)

When we collect feedbacks from users, it is possible that an issue becomes a megaissue that exposes manifold subproblems. Such issue requires multiple commits and potentially long time to resolve, leaving the megaissue hanging. This cannot reflexts

### case study

[This issue (#6)](https://github.com/szcompressor/cuSZ/issues/6) exposes at least 6 subproblems,

1. Zip and unzip is not decoupled, resolved in (part of [v0.1.1](https://github.com/szcompressor/cuSZ/releases/tag/v0.1.1)).
2. Not able to specify output path, resolved (part of [v0.1.1](https://github.com/szcompressor/cuSZ/releases/tag/v0.1.1)).
3. Pascal GPU failed because of overestimated cache size, resolved in [037bf6e](https://github.com/szcompressor/cuSZ/commit/037bf6e6afd01e684c40f439bc5e78f3b3b02cb3).
4. It is after a while that we found big-endian is input, as of 11/5/2020, there is no input addon to convert big-endian to little-endian.
5. It is after a while that we determine the platform, so it would be convinient to attach machine information, resolved in [31a2962](https://github.com/szcompressor/cuSZ/commit/31a2962bf50685b275f751c3750e6c35e0b96695).
6. Large Parihaka datum (5 GiB) runs out of GPU memory; although it's not exposed by 16-GiB P100 or other target devices, solved in [90b521b](https://github.com/szcompressor/cuSZ/commit/90b521b30925e42ede5a635111b123f6238b0e0e) (refer to [issue #19](https://github.com/szcompressor/cuSZ/issues/19)).
7. (This megaissue is still in development.)


### template
title
```
[log] subproblem description
```
header (copy and paste into codeblock of markdown)
```
(v0) This record reflects early-stage development and is subsequent to commited code. 
(v0) There are several reasons that this record shows up,
   1. An issue from user or developer exposes manifold subproblems.
   2. Such a record remarks a solved subproblem.
   3. The solved subproblem is not instaneously documented.
   4. This is helpful as a cross-reference.
```

A full document can be
```
[title]
----- body below ----
[header]
subproblem descrption with commit SHA
```

### how to use

1. Open a new issue.
2. Use the template above: use the title format, and put the header in the begining of issue description that contains commit SHA such that the megaissue is referenced that this log shows up in the megaissue.
4. Close the issue.
5. After megaissue is eventually resolved, a summary is preferable, which contans a list of solved subproblems in terms of a list of such log-issues.


