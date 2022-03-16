theory Oracle
  imports HOL.Rat
begin
(*
  This is a little Isabelle theory file I made to double check my algebra so I don't waste time trying to test
  properties that aren't true. Each lemma represents a property and the lines underneath are proofs. I have named the
  lemmas according to the abstract test in causal_tests.json that they relate to.
*)

(*
  This shows that, when we increase w or h by c, we expect to get 2ic more lines in the space.
*)
lemma num_lines_width_height: "((2::rat) * i * (w + h + c)) - (2 * i * (w + h)) = 2 * i * c"
  by algebra

(*
  This shows that, when we increase i by c, we expect to get 2c(h+w) more lines in the space.
*)
lemma intensity_num_lines_abs: "((2::rat) * (i+c) * (w + h)) - (2 * i * (w + h)) = 2 * c * (h + w)"
  by algebra

(*
  This shows that, when we increase w or h by c, we expect to see less lines per unit area.
*)
lemma num_lines_width_height_unit:
  assumes "w>0" and "h>0" and "c > 0" and "i > 0"
  shows "((2::rat) * i * (w + h + c))/((w+c)*h) < (2 * i * (w + h))/(w*h)"
proof-
  have aux1: "\<And>x x' y. x > 0 \<Longrightarrow> y > 0 \<Longrightarrow> (x::rat) * y < x' * y \<Longrightarrow> x < x'"
    by simp
  have aux2: "\<And>x y z. ((x::rat)*y)/z = x*(y/z)"
    by simp
  have aux3: "\<And>x x'. ((w::rat)/w)*x < x' \<Longrightarrow> x < x'"
    by (simp add: assms(1) less_imp_not_eq2)
  have aux3a: "\<And>x x'. x < x' * ((w + c) / (w + c)) \<Longrightarrow> x < x'"
    by (simp add: add_pos_pos assms(1) assms(3) less_imp_not_eq2)
  have aux5: "\<And> x y z. (y+z)*(x::rat) = (x*y)+(x*z)"
    by algebra
  show ?thesis
    apply (insert assms)
  apply (simp only: aux2)
  apply (rule Rings.linordered_comm_semiring_strict_class.comm_mult_strict_left_mono[of _ _ "2*i"])
   prefer 2 apply simp
  apply (rule aux1[of _ h])
    apply simp
   apply simp
  apply simp
  apply (rule aux3)
    apply (simp only: Fields.field_class.times_divide_times_eq)
  apply (rule aux3a)
  apply (simp only: Fields.field_class.times_divide_times_eq)
  apply (rule Fields.linordered_field_class.divide_strict_right_mono)
   apply (simp add: Groups.algebra_simps(18) aux5)
    by simp
qed



end
