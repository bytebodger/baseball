import pg from 'pg';
import { postgresConfig } from './postgresConfig.js';

export const dbClient = new pg.Client(postgresConfig);