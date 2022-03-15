theory Oracle
  imports HOL.Rat
begin

(*
  This shows that, when we increase w or h by c, we expect to get 2ic more lines in the space.
*)
lemma "((2::rat) * i * (w + h + c)) - (2 * i * (w + h)) = 2 * i * c"
  by algebra

lemma aux1: "x > 0 \<Longrightarrow> y > 0 \<Longrightarrow> (x::rat) * y < x' * y \<Longrightarrow> x < x'"
  by simp

lemma aux2: "((x::rat)*y)/z = x*(y/z)"
  by simp

lemma aux3: "w \<noteq> 0 \<Longrightarrow> ((w::rat)/w)*x < x' \<Longrightarrow> x < x'"
  by simp

lemma aux3a: "w \<noteq> 0 \<Longrightarrow> x < x'*((w::rat)/w) \<Longrightarrow> x < x'"
  by simp

lemma aux5: "(y+z)*(x::rat) = (x*y)+(x*z)"
  by algebra

(*
  This shows that, when we increase w or h by c, we expect to see less lines per unit area.
*)
lemma "w>0 \<Longrightarrow> h>0 \<Longrightarrow> c > 0 \<Longrightarrow> i > 0 \<Longrightarrow> ((2::rat) * i * (w + h + c))/((w+c)*h) < (2 * i * (w + h))/(w*h)"
  apply (simp only: aux2)
  apply (rule Rings.linordered_comm_semiring_strict_class.comm_mult_strict_left_mono[of _ _ "2*i"])
   prefer 2 apply simp
  apply (rule aux1[of _ h])
    apply simp
   apply simp
  apply simp
  apply (rule aux3[of w])
   apply simp
  apply (simp only: Fields.field_class.times_divide_times_eq)
  apply (rule aux3a[of "w+c"])
   apply simp
  apply (simp only: Fields.field_class.times_divide_times_eq)
  apply (rule Fields.linordered_field_class.divide_strict_right_mono)
   apply (simp add: Groups.algebra_simps(18) aux5)
  by simp



end
