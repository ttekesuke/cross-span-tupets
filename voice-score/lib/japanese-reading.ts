export type JapaneseTokenInput = {
  surface_form: string;
  pronunciation?: string;
  reading?: string;
  pos?: string;
};

export type PronounceableJapaneseToken = {
  surface: string;
  pronunciation: string;
  reading: string;
  pos: string;
};

const DIGITS = ["ゼロ", "イチ", "ニ", "サン", "ヨン", "ゴ", "ロク", "ナナ", "ハチ", "キュウ"];
const LARGE_UNITS = ["", "マン", "オク", "チョウ", "ケイ"];

function underTenThousand(value: number) {
  if (!value) return "";
  const thousands = Math.floor(value / 1000);
  const hundreds = Math.floor((value % 1000) / 100);
  const tens = Math.floor((value % 100) / 10);
  const ones = value % 10;
  let reading = "";

  if (thousands) {
    if (thousands === 3) reading += "サンゼン";
    else if (thousands === 8) reading += "ハッセン";
    else reading += `${thousands === 1 ? "" : DIGITS[thousands]}セン`;
  }
  if (hundreds) {
    if (hundreds === 3) reading += "サンビャク";
    else if (hundreds === 6) reading += "ロッピャク";
    else if (hundreds === 8) reading += "ハッピャク";
    else reading += `${hundreds === 1 ? "" : DIGITS[hundreds]}ヒャク`;
  }
  if (tens) reading += `${tens === 1 ? "" : DIGITS[tens]}ジュウ`;
  if (ones) reading += DIGITS[ones];
  return reading;
}

/** Arabic numerals emitted by Whisper are not given a reading by IPADIC. */
export function numberToKatakana(value: string) {
  const normalized = value.normalize("NFKC").replaceAll(",", "");
  const match = normalized.match(/^([+-]?)(\d+)(?:\.(\d+))?$/u);
  if (!match) return "";
  const [, sign, integerPart, decimalPart] = match;
  const signReading = sign === "-" ? "マイナス" : sign === "+" ? "プラス" : "";

  let integerReading = "";
  if (integerPart.length > 1 && integerPart.startsWith("0")) {
    integerReading = Array.from(integerPart, (digit) => DIGITS[Number(digit)]).join("");
  } else if (integerPart.length > LARGE_UNITS.length * 4) {
    integerReading = Array.from(integerPart, (digit) => DIGITS[Number(digit)]).join("");
  } else {
    const padded = integerPart.padStart(Math.ceil(integerPart.length / 4) * 4, "0");
    const groups = padded.match(/.{4}/gu) ?? [];
    groups.forEach((group, index) => {
      const amount = Number(group);
      if (!amount) return;
      const unitIndex = groups.length - index - 1;
      integerReading += underTenThousand(amount) + LARGE_UNITS[unitIndex];
    });
    if (!integerReading) integerReading = DIGITS[0];
  }

  const decimalReading = decimalPart
    ? `テン${Array.from(decimalPart, (digit) => DIGITS[Number(digit)]).join("")}`
    : "";
  return signReading + integerReading + decimalReading;
}

function hiraganaToKatakana(value: string) {
  return value.replace(/[ぁ-ゖ]/gu, (character) =>
    String.fromCharCode(character.charCodeAt(0) + 0x60));
}

function validDictionaryReading(value: string | undefined) {
  return value && value !== "*" ? value : "";
}

function fallbackReading(surface: string) {
  const numeric = numberToKatakana(surface);
  if (numeric) return numeric;
  return /^[ぁ-ゖァ-ヶー]+$/u.test(surface) ? hiraganaToKatakana(surface) : "";
}

/**
 * Produce the single token stream shared by phonemization, alignment and score
 * lyrics. Unknown kana and Arabic numerals must not silently disappear.
 */
export function normalizeJapaneseTokens(tokens: JapaneseTokenInput[]): PronounceableJapaneseToken[] {
  const normalizedSurfaces = tokens.map((token) => token.surface_form.normalize("NFKC"));
  return tokens.flatMap((token, index) => {
    const surface = normalizedSurfaces[index];
    let reading = validDictionaryReading(token.pronunciation)
      || validDictionaryReading(token.reading)
      || fallbackReading(surface);

    const previousIsNumber = index > 0 && /^\d+$/u.test(normalizedSurfaces[index - 1]);
    if (previousIsNumber && surface === "月") reading = "ガツ";
    if (previousIsNumber && surface === "日") reading = "ニチ";
    reading = hiraganaToKatakana(reading.normalize("NFKC"));
    if (!/[ァ-ヶ]/u.test(reading)) return [];

    return [{
      surface,
      pronunciation: reading,
      reading,
      pos: token.pos ?? "",
    }];
  });
}
