import random
import unittest

import utf8_to_symato as m


class TestUtf8ToSymato(unittest.TestCase):
    def test_smoke_main_api(self):
        self.assertEqual(m.text_to_symato("Việt Nam"), "^viet|zj ^nam|")
        self.assertEqual(m.text_to_symato("Đường đi"), "^dduong|wf ddi|")

    def test_keep_non_vn_preserves_whitespace(self):
        s = "a\t  \n\r\nb"
        out = m.text_to_symato(s, keep_non_vn=True)
        self.assertEqual(out, "a|\t  \n\r\nb")

        s2 = " \t\n  "
        out2 = m.text_to_symato(s2, keep_non_vn=True)
        self.assertEqual(out2, s2)

    def test_keep_non_vn_false_drops_non_letters(self):
        # When keep_non_vn=False, punctuation/whitespace are dropped.
        self.assertEqual(m.text_to_symato("(Việt)", keep_non_vn=False), "^viet|zj")
        self.assertEqual(m.text_to_symato("Việt Nam", keep_non_vn=False), "^viet|zj^nam|")

    def test_boundary_rules(self):
        # Blocking boundaries: token after these should not convert
        self.assertEqual(m.text_to_symato("email@gmail.com"), "email@gmail.com")
        self.assertEqual(m.text_to_symato("gmail.com"), "gmail.com")
        self.assertEqual(m.text_to_symato("/Việt"), "/Việt")
        self.assertEqual(m.text_to_symato("\\Việt"), "\\Việt")
        self.assertEqual(m.text_to_symato(".Việt"), ".Việt")
        self.assertEqual(m.text_to_symato("@Việt"), "@Việt")

        # Allowing boundaries
        self.assertEqual(m.text_to_symato('(Việt)'), '(^viet|zj)')
        self.assertEqual(m.text_to_symato('"Việt"'), '"^viet|zj"')

    def test_caps_prefix(self):
        self.assertEqual(m.text_to_symato("Việt"), "^viet|zj")
        self.assertEqual(m.text_to_symato("VIỆT"), "^^viet|zj")
        self.assertEqual(m.text_to_symato("Đi"), "^ddi|")
        self.assertEqual(m.text_to_symato("ĐI"), "^^ddi|")

    def test_syllable_to_symato_contract(self):
        sym, mt, up1, upall = m.syllable_to_symato("Việt")
        self.assertEqual((sym, mt, up1, upall), ("viet", "|zj", True, False))

        sym, mt, up1, upall = m.syllable_to_symato("VIỆT")
        self.assertEqual((sym, mt, up1, upall), ("viet", "|zj", False, True))

    def test_symato_to_telex(self):
        self.assertEqual(m.symato_to_telex("nguoi", "|wf"), "nguoiwf")
        self.assertEqual(m.symato_to_telex("ma", "|"), "ma")
        self.assertEqual(m.symato_to_telex("ma", "|s"), "mas")

    def test_no_crash_random(self):
        alphabet = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
            "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
            " !?.,;:-_()[]{}\"'@/\\\n\t\r"
        )

        rnd = random.Random(0)
        for _ in range(2000):
            n = rnd.randint(0, 200)
            s = "".join(rnd.choice(alphabet) for _ in range(n))
            out = m.text_to_symato(s, keep_non_vn=True)

            # whitespace preserved
            self.assertEqual(
                "".join(c for c in s if c.isspace()),
                "".join(c for c in out if c.isspace()),
            )


if __name__ == "__main__":
    unittest.main()
