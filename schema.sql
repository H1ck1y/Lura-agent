CREATE DATABASE IF NOT EXISTS luya_shop
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE luya_shop;

CREATE TABLE IF NOT EXISTS products (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  name VARCHAR(160) NOT NULL,
  species VARCHAR(40) NOT NULL,
  water_type VARCHAR(40) NOT NULL,
  bait_type VARCHAR(60) NOT NULL,
  brand VARCHAR(80) NOT NULL,
  price DECIMAL(10, 2) NOT NULL,
  stock INT UNSIGNED NOT NULL DEFAULT 0,
  description TEXT NOT NULL,
  PRIMARY KEY (id),
  KEY idx_products_species (species),
  KEY idx_products_price_stock (price, stock),
  CONSTRAINT chk_products_price_nonnegative CHECK (price >= 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS species_aliases (
  alias VARCHAR(40) NOT NULL,
  canonical_species VARCHAR(40) NOT NULL,
  PRIMARY KEY (alias),
  KEY idx_species_aliases_canonical (canonical_species)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
