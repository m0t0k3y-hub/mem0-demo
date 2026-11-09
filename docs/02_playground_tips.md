#### 試してみよう

1. **add** をそのまま実行 → 会話がどんな「事実」に分解されたかを見る
2. 同じ内容をもう一度 add → 重複として追加されないことを確認
3. `infer` を外して add → 文字列がそのまま 1 件で入る
4. **search** で `explain` を ON → `semantic_score` などスコアの内訳を見る
5. threshold を 0 / 0.3 / 0.5 と変えて、ヒット件数の変化を見る
6. filters に `{"user_id": "suzuki", "category": "profile"}` のようなメタデータ条件を足す
7. **管理** タブで行を選び、update → history で変更履歴を確認

#### filters の書き方

```python
{"user_id": "u1"}                               # スコープ
{"user_id": "u1", "run_id": "s1"}               # 組み合わせ（AND）
{"user_id": "u1", "category": "food"}           # メタデータ完全一致
{"user_id": "u1", "importance": {"gte": 3}}     # 比較演算子
{"user_id": "u1", "tag": {"in": ["a", "b"]}}    # in / nin
```
