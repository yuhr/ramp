// Copyright 2015 The Ramp Developers
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//#![feature(label_break_value)]
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

/*!
 * Base conversion utilities
 *
 * Provides functions for converting an integer to/from a given base. In both `to_base` and
 * `from_base` the base-N output or input (respectively) is stored as raw bytes. That means that a
 * base-10 input contains bytes each with a value from 0-9.
 */

use std::intrinsics::assume;

use ll;
use ll::limb::Limb;
use ll::limb_ptr::{Limbs, LimbsMut};

/// Information for converting to/from a given base, B. Stored in a table generated
/// by build.rs
struct Base {
  /// Number of digits a limb can hold in a given base. Except if B is a power of 2,
  /// in which case it is equal to the number of bits per digit in base B
  digits_per_limb: u32,
  /// The "big base", essentially the largest B^m that will fit in a limb.
  /// Should be the same as B^digits_per_limb
  big_base: Limb
}
// Include BASES table
include!(concat!(env!("OUT_DIR"), "/bases_table.rs"));

#[inline(always)]
fn div_unnorm(n: Limb, d: Limb) -> (Limb, Limb) {
  (n / d, n % d)
}

#[inline]
/// Returns the number of digits needed to represent `p` in base `base`
/// without sign. If the base is not a power of two, the result is only
/// an estimate. It can equal the the actually needed digits or overestimate
/// by 1.
/// Returns 1 if the number is 0;
pub unsafe fn num_base_digits(p: Limbs, n: i32, base: u32) -> usize {
  debug_assert!(base >= 2);
  assume(base >= 2);

  if n == 0 {
    return 1;
  }

  let cnt = (*p.offset((n - 1) as isize)).leading_zeros() as usize;
  let total_bits = (Limb::BITS * (n as usize)) - cnt;

  if base == 2 {
    // no need to do anything complicated here at all, so let's go
    // as fast as possible (this is a somewhat common case, due to
    // `bit_length`)
    total_bits
  } else if base.is_power_of_two() {
    let bits_per_digit = BASES.get_unchecked(base as usize).big_base.0 as usize;
    if bits_per_digit.is_power_of_two() {
      // doing an actual division here is much slower than this
      (total_bits + bits_per_digit - 1) >> bits_per_digit.trailing_zeros()
    } else {
      (total_bits + bits_per_digit - 1) / bits_per_digit
    }
  } else {
    // Not sure if using floating-point arithmetic here is the best idea,
    // but it should be a reasonable accurate result, maybe a little high
    let total_bits = total_bits as f64;

    let lg2b = (base as f64).log2();
    let digits = total_bits / lg2b;
    return digits.ceil() as usize;
  }
}

#[inline]
pub fn base_digits_to_len(num: usize, base: u32) -> usize {
  debug_assert!(base >= 2);

  if num == 0 {
    return 0;
  }

  let digits_per_limb = BASES[base as usize].digits_per_limb as usize;

  (num / digits_per_limb) + 1
}

/**
 * Converts `nn` limbs at `np` to the given base, storing the output in `out`. `out` is assumed to
 * have enough space for the entire digit. The output is stored from most-significant digit to least.
 *
 * The values in `out` are the raw values of the base. Conversion for output should be done as a second
 * step.
*/
pub unsafe fn to_base<F: FnMut(u8)>(
  base: u32,
  np: Limbs,
  nn: i32,
  mut out_byte: F,
  len: Option<usize>
) {
  debug_assert!(nn >= 0);
  debug_assert!(base < BASES.len() as u32);
  debug_assert!(base >= 2);
  assume(base < BASES.len() as u32);
  assume(base >= 2);

  let mut count = 0;

  if nn <= 0 {
    // Emit at least one zero
    out_byte(0);
    count += 1;
    if let Some(len) = len {
      while count < len {
        out_byte(0);
        count += 1;
      }
    }
    return;
  }

  // Fast path for powers-of-two, since each limb is already in base B^m format
  if base.is_power_of_two() {
    let bits_per_digit = BASES.get_unchecked(base as usize).big_base.0 as usize;
    assume(bits_per_digit > 0);

    let mut n1 = *np.offset((nn - 1) as isize);

    let cnt = n1.leading_zeros() as usize;
    let mut bits = Limb::BITS * (nn as usize) - cnt;
    let cnt = bits % bits_per_digit;
    if cnt != 0 {
      bits += bits_per_digit - cnt;
    }

    let mut bit_pos: isize = (bits - (nn as usize - 1) * Limb::BITS) as isize;

    let mut i = nn - 1;
    // Convert each limb by shifting and masking to get the value for each output digit
    loop {
      bit_pos -= bits_per_digit as isize;
      while bit_pos >= 0 {
        let b = ((n1 >> (bit_pos as usize)) & ((Limb(1) << bits_per_digit) - 1)).0 as u8;
        out_byte(b);
        count += 1;
        if let Some(len) = len {
          if len == count {
            println!("{}, {}", count, len);
            return;
          }
        }
        bit_pos -= bits_per_digit as isize;
      }
      i -= 1;
      if i < 0 {
        break;
      }

      let n0 = (n1 << ((-bit_pos) as usize)) & ((Limb(1) << bits_per_digit) - 1);
      n1 = *np.offset(i as isize);
      bit_pos += Limb::BITS as isize;
      // We have a potential overlap of bits, so get the value of the digit
      // that spans the two limbs.
      // The specific situation is (using 8-bit limbs as demonstration):
      //
      //    bbbbbbbb bbbbbbbb
      //          ^---^         Bits for next digit
      let b = (n0 | (n1 >> (bit_pos as usize))).0 as u8;
      out_byte(b);
      count += 1;
      if let Some(len) = len {
        if len == count {
          println!("{}, {}", count, len);
          return;
        }
      }
    }

    if let Some(len) = len {
      while count < len {
        out_byte(0);
        count += 1;
      }
    }
  } else {
    // TODO: Use divide-and-conquer for large numbers
    to_base_impl(len, base, np, nn, out_byte);
  }
}

pub unsafe fn to_bytes_le<F: FnMut(u8)>(np: Limbs, nn: i32, mut out_byte: F, len: Option<usize>) {
  let bits_per_digit = BASES.get_unchecked(256).big_base.0 as usize;
  let mut count = 0;

  let mask_one_digit = (Limb(1) << bits_per_digit) - 1;

  // Convert each limb by shifting and masking to get the value for each output digit
  for i in 0..nn {
    let n1 = *np.offset(i as isize);
    for bit_pos in (0..Limb::BITS).step_by(bits_per_digit) {
      let b = ((n1 >> (bit_pos as usize)) & mask_one_digit).0 as u8;
      out_byte(b);
      count += 1;
      if let Some(len) = len {
        if count == len {
          return;
        }
      }
    }
  }
  if let Some(len) = len {
    while count < len {
      out_byte(0);
      count += 1;
    }
  }
}

unsafe fn to_base_impl<F: FnMut(u8)>(
  len: Option<usize>,
  base: u32,
  np: Limbs,
  mut nn: i32,
  mut out_byte: F
) {
  debug_assert!(2 < base);

  let buf_len = num_base_digits(np, nn, base);
  let mut buf: Vec<u8> = vec![0; buf_len];
  let mut r: Vec<Limb> = vec![Limb(0); (nn + 1) as usize];
  let rp = LimbsMut::new(&mut r[0], 0, r.len() as i32);

  ll::copy_incr(np, rp.offset(1), nn);

  let s: *mut u8 = &mut buf[0];
  let mut sp = s.offset(buf_len as isize);

  let base = Limb(base as ll::limb::BaseInt);
  let mut count = 0;

  'outer: {
    let Base {
      digits_per_limb,
      big_base
    } = BASES.get_unchecked(base.0 as usize);

    // Process limbs from least-significant to most, until there is only one
    // limb left
    while 1 < nn {
      // Divide rp by the big_base, with a single fractional limb produced.
      // The fractional limb is approximately 1/remainder
      ll::divrem_1(rp, 1, rp.offset(1).as_const(), nn, *big_base);

      nn -= if *rp.offset(nn as isize) == 0 { 1 } else { 0 };
      let mut frac = *rp + 1;
      // The loop below produces digits from most-significant to least, but
      // the containing loop works from least signficant limb up, so move
      // the first position for the output for this limb. Since we know
      // there is at least one more limb to process after this one, it's
      // safe to output all digits that may be produced.
      sp = sp.offset(-(*digits_per_limb as isize));
      for _ in 0..*digits_per_limb {
        // Multiply the fraction from divrem by the base, the overflow
        // amount is the next digit we want
        let (digit, f) = frac.mul_hilo(base);
        frac = f;
        *sp = digit.0 as u8;
        sp = sp.offset(1);

        count += 1;
        // if let Some(len) = len {
        //   if count == len {
        //     break 'outer;
        //   }
        // }
      }

      sp = sp.offset(-(*digits_per_limb as isize));
    }

    // Last limb, use normal conversion for this one so we
    // don't overshoot the number of digits
    let mut ul = *rp.offset(1);
    while ul != 0 {
      let (q, r) = div_unnorm(ul, base);
      sp = sp.offset(-1);
      *sp = r.0 as u8;
      ul = q;

      count += 1;
      // if let Some(len) = len {
      //   if count == len {
      //     break 'outer;
      //   }
      // }
    }
  }

  // Copy the temporary buffer into the output string
  sp = s.offset(0);
  for i in 0..count {
    if let Some(len) = len {
      if i == len {
        break;
      }
    }
    out_byte(*sp);
    sp = sp.offset(1);
  }

  // Output any padding zeros we may want
  if let Some(len) = len {
    for _ in count..len {
      out_byte(0);
    }
  }
}

/**
 * Converts the base `base` bytestring {bp, bs}, storing the limbs in `out`. `out` is assumed to
 * have enough space to store the result.
*/
pub unsafe fn from_base(mut out: LimbsMut, bp: *const u8, bs: i32, base: u32) -> usize {
  debug_assert!(bs > 0);
  debug_assert!(base < BASES.len() as u32);
  debug_assert!(base >= 2);
  assume(base < BASES.len() as u32);
  assume(base >= 2);

  if bs <= 0 {
    *out = Limb(0);
    return 1;
  }

  if base.is_power_of_two() {
    let bits_per_digit = BASES.get_unchecked(base as usize).big_base.0 as usize;
    assume(bits_per_digit > 0);

    let mut size = 0;

    let mut b = bp.offset((bs - 1) as isize);
    let mut res_digit = Limb(0);
    let mut next_bitpos = 0;
    while b >= bp {
      let digit = Limb((*b) as ll::limb::BaseInt);

      res_digit = res_digit | (digit << next_bitpos);
      next_bitpos += bits_per_digit;
      if next_bitpos >= Limb::BITS {
        next_bitpos -= Limb::BITS;
        *out.offset(size as isize) = res_digit;
        size += 1;
        res_digit = digit >> (bits_per_digit - next_bitpos);
      }

      b = b.offset(-1);
    }

    if res_digit > 0 {
      *out.offset(size as isize) = res_digit;
      size += 1;
    }

    return size;
  }

  // TODO, use a divide-and-conquer algorithm for large inputs
  from_base_small(out, bp, bs, base)
}

unsafe fn from_base_small(mut out: LimbsMut, mut bp: *const u8, bs: i32, base: u32) -> usize {
  debug_assert!(base > 2);
  assume(base > 2);

  let big_base = BASES.get_unchecked(base as usize).big_base;
  let digits_per_limb = BASES.get_unchecked(base as usize).digits_per_limb;

  let mut i = digits_per_limb;
  let mut size: usize = 0;
  while i < (bs as u32) {
    let mut res_digit = Limb((*bp) as ll::limb::BaseInt);
    bp = bp.offset(1);

    if base == 10 {
      let mut j = digits_per_limb - 1;
      while j > 0 {
        res_digit = res_digit * 10 + ((*bp) as ll::limb::BaseInt);
        bp = bp.offset(1);
        j -= 1;
      }
    } else {
      let mut j = digits_per_limb - 1;
      while j > 0 {
        res_digit = res_digit * (base as ll::limb::BaseInt) + ((*bp) as ll::limb::BaseInt);
        bp = bp.offset(1);
        j -= 1;
      }
    }

    if size == 0 {
      if res_digit != 0 {
        *out = res_digit;
        size = 1;
      }
    } else {
      let mut carry = ll::mul_1(out, out.as_const(), size as i32, big_base);
      carry = carry + ll::add_1(out, out.as_const(), size as i32, res_digit);
      if carry != 0 {
        *out.offset(size as isize) = carry;
        size += 1;
      }
    }

    i += digits_per_limb;
  }

  let mut big_base = base as ll::limb::BaseInt;
  let mut res_digit = Limb((*bp) as ll::limb::BaseInt);
  bp = bp.offset(1);

  if base == 10 {
    let mut j = (bs as u32) - (i - digits_per_limb) - 1;
    while j > 0 {
      res_digit = res_digit * 10 + ((*bp) as ll::limb::BaseInt);
      big_base *= 10;
      bp = bp.offset(1);
      j -= 1;
    }
  } else {
    let mut j = (bs as u32) - (i - digits_per_limb) - 1;
    while j > 0 {
      res_digit = res_digit * (base as ll::limb::BaseInt) + ((*bp) as ll::limb::BaseInt);
      bp = bp.offset(1);
      big_base *= base as ll::limb::BaseInt;
      j -= 1;
    }
  }

  if size == 0 {
    if res_digit != 0 {
      *out = res_digit;
      size = 1;
    }
  } else {
    let mut carry = ll::mul_1(out, out.as_const(), size as i32, Limb(big_base));
    carry = carry + ll::add_1(out, out.as_const(), size as i32, res_digit);
    if carry != 0 {
      *out.offset(size as isize) = carry;
      size += 1;
    }
  }

  return size;
}