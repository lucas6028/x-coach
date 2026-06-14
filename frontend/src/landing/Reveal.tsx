import { type ReactNode } from "react";
import { motion, useReducedMotion } from "motion/react";

interface Props {
  children: ReactNode;
  className?: string;
  delay?: number;
  y?: number;
}

// Single scroll-reveal primitive used across the landing page. Honors
// prefers-reduced-motion by rendering content statically (no transform).
export default function Reveal({ children, className, delay = 0, y = 22 }: Props) {
  const reduce = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial={reduce ? false : { opacity: 0, y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.3 }}
      transition={{ duration: 0.6, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.div>
  );
}
